"""Bounded PPO runs, versioned artifacts, and separate deterministic evaluation."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import hashlib
import io
import json
import time
import torch

from .algorithm import ActorCritic, PPO, PPOConfig, generalized_advantage
from .curriculum import next_stage
from .evaluation import CompletionAudit, ProgressAudit
from .goals import GOAL_MODES
from .multigoal import contract, case_name
from .goal_curriculum import MODES as GOAL_CURRICULA, bounds as goal_bounds, progress as goal_progress, state as goal_curriculum_state


FORMAT_VERSION = 1
TASK_VERSION = "crx_full_pick_place_v2"


def control_settings(args):
    return {"gravity_compensation": getattr(args, "gravity_compensation", False),
            "gripper_damping": getattr(args, "gripper_damping", None),
            "gripper_max_velocity": getattr(args, "gripper_max_velocity", None),
            "solver_velocity_iterations": getattr(args, "solver_velocity_iterations", None),
            "physics_hz": getattr(args, "physics_hz", 60),
            "success_bonus": getattr(args, "success_bonus", 100.),
            "reward_scale": getattr(args, "reward_scale", 1.),
            "completion_hold_steps": getattr(args, "completion_hold_steps", 10),
            "completion_retry_mode": getattr(args, "completion_retry_mode", "retry")}


def saved_control_setting(data, key):
    default = {"gravity_compensation": False, "physics_hz": 60, "success_bonus": 100., "reward_scale": 1.,
               "completion_hold_steps": 10, "completion_retry_mode": "retry"}.get(key)
    return data.get(key, default)


def task_version(env):
    if getattr(env.args, "task", "full") == "staged_whole":
        return "crx_whole_pick_place_v6"
    if getattr(env.args, "task", "full") == "staged_precision":
        return "crx_staged_pick_place_v5"
    if getattr(env.args, "task", "full") == "staged_hold":
        return "crx_staged_pick_place_v4"
    return "crx_staged_pick_place_v3" if getattr(env.args, "task", "full") == "staged" else TASK_VERSION


def save_checkpoint(path, model, learner, env, iteration, cfg):
    payload = {
        "format_version": FORMAT_VERSION, "task_version": task_version(env),
        "algorithm": "PPO", "teacher_samples": 0,
        "goal_mode": getattr(env.args, "goal_mode", "curriculum"),
        "goal_contract": contract(getattr(env.args, "goal_mode", "curriculum")),
        "goal_curriculum": goal_curriculum_state(env),
        "task_transfer": getattr(env, "task_transfer", None),
        "obs_dim": env.obs_dim, "action_dim": env.action_dim,
        "control": env.args.control,
        "gravity_compensation": getattr(env.args, "gravity_compensation", False),
        "gripper_damping": getattr(env.args, "gripper_damping", None),
        "gripper_max_velocity": getattr(env.args, "gripper_max_velocity", None),
        "solver_velocity_iterations": getattr(env.args, "solver_velocity_iterations", None),
        "physics_hz": getattr(env.args, "physics_hz", 60),
        "success_bonus": getattr(env.args, "success_bonus", 100.),
        "reward_scale": getattr(env.args, "reward_scale", 1.),
        "completion_hold_steps": getattr(env.args, "completion_hold_steps", 10),
        "completion_retry_mode": getattr(env.args, "completion_retry_mode", "retry"),
        "initialization": getattr(env, "initialization", {"kind": "random"}),
        "model": model.state_dict(), "optimizer": learner.optimizer.state_dict(),
        "iteration": iteration, "curriculum": env.curriculum, "ppo_config": asdict(cfg),
        "stage": getattr(env, "task_stage", "full"),
        "torch_rng": torch.get_rng_state(), "env_rng": env.rng.get_state(),
        "cuda_rng": torch.cuda.get_rng_state(env.device) if env.device.type == "cuda" else None,
        "reset_q": env.reset_q[0].cpu(),
        "reset_rotation": env.reset_rotation[0].cpu() if hasattr(env, "reset_rotation") else None,
        "usd_sha256": hashlib.sha256(env.args.usd.read_bytes()).hexdigest(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, learner, env, resume):
    checkpoint_bytes = path.read_bytes()
    data = torch.load(io.BytesIO(checkpoint_bytes), map_location=env.device, weights_only=True)
    if data.get("format_version") != FORMAT_VERSION or data.get("task_version") != task_version(env):
        raise ValueError("Incompatible full-task PPO checkpoint")
    if data["obs_dim"] != env.obs_dim or data["action_dim"] != env.action_dim:
        raise ValueError("Checkpoint observation/action dimensions do not match")
    if data.get("control", "joint") != env.args.control:
        raise ValueError("Checkpoint control mode does not match --control")
    source_goal = data.get("goal_mode", "curriculum")
    target_goal = getattr(env.args, "goal_mode", "curriculum")
    retarget = getattr(env.args, "retarget", False)
    if source_goal not in GOAL_MODES or target_goal not in GOAL_MODES:
        raise ValueError("Unknown checkpoint or requested goal mode")
    if source_goal == "multi_goal" and data.get("goal_contract") != contract(source_goal):
        raise ValueError("Multi-goal checkpoint task contract differs")
    if source_goal != target_goal:
        if not (resume and retarget and data.get("algorithm") == "PPO" and data.get("teacher_samples") == 0):
            raise ValueError("Checkpoint goal differs; use explicit --retarget for training, never relabel evaluation")
    elif retarget:
        raise ValueError("--retarget requires a different goal mode")
    saved_curriculum = data.get("goal_curriculum")
    saved_mode = saved_curriculum["mode"] if saved_curriculum else "none"
    requested_mode = getattr(env.args, "goal_curriculum", "none")
    if saved_mode not in GOAL_CURRICULA or requested_mode not in GOAL_CURRICULA:
        raise ValueError("Unsupported destination curriculum")
    if resume and saved_mode != requested_mode and not (retarget and saved_mode == "none"):
        raise ValueError("Destination curriculum differs; resume must preserve its saved mode")
    if saved_curriculum:
        if source_goal != "multi_goal" or saved_curriculum["sampling_region"] != goal_bounds(saved_curriculum["level"]):
            raise ValueError("Invalid saved destination curriculum")
    env.goal_curriculum_mode = requested_mode if resume else saved_mode
    env.goal_curriculum_level = saved_curriculum["level"] if saved_curriculum else 0.
    differences = {key: {"checkpoint": saved_control_setting(data, key), "evaluation": value}
                   for key, value in control_settings(env.args).items()
                   if saved_control_setting(data, key) != value}
    if differences and (resume or not getattr(env.args, "eval_control_ablation", False)):
        raise ValueError(f"Checkpoint control settings do not match: {differences}")
    env.control_ablation = differences
    if resume and data.get("ppo_config") != asdict(learner.cfg):
        raise ValueError("PPO settings differ from the saved optimizer run")
    if data["usd_sha256"] != hashlib.sha256(env.args.usd.read_bytes()).hexdigest():
        raise ValueError("USD differs from the checkpoint's training asset")
    model.load_state_dict(data["model"])
    env.loaded_checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    env.task_transfer = ({"kind": "goal_retarget", "source_goal_mode": source_goal,
                          "goal_mode": target_goal, "source_sha256": env.loaded_checkpoint_sha256,
                          "ppo_optimizer_rng_restored": True} if retarget else data.get("task_transfer"))
    env.curriculum = float(data["curriculum"])
    env.task_stage = data.get("stage", "full")
    env.initialization = data.get("initialization", {"kind": "random"})
    env.reset_q[:] = data["reset_q"].to(env.device)
    if data.get("reset_rotation") is not None:
        env.reset_rotation[:] = data["reset_rotation"].to(env.device)
    if resume:
        learner.optimizer.load_state_dict(data["optimizer"])
        torch.set_rng_state(data["torch_rng"].cpu())
        env.rng.set_state(data["env_rng"].cpu())
        if env.device.type == "cuda" and data.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(data["cuda_rng"].cpu(), env.device)
    # Physics episode state is intentionally not restored: resume starts new episodes.
    return int(data["iteration"])


def initialize_actor(path, model, env):
    """Explicit transfer between compatible PPO experiments, not optimizer resume."""
    checkpoint_bytes = path.read_bytes()
    data = torch.load(io.BytesIO(checkpoint_bytes), map_location=env.device, weights_only=True)
    if data.get("algorithm") != "PPO" or data.get("teacher_samples") != 0:
        raise ValueError("Actor initialization requires a teacher-free PPO checkpoint")
    if data.get("task_version") not in (TASK_VERSION, "crx_staged_pick_place_v3", "crx_staged_pick_place_v4", "crx_staged_pick_place_v5", "crx_whole_pick_place_v6"):
        raise ValueError("Unknown actor observation/action contract")
    if data.get("obs_dim") != env.obs_dim or data.get("action_dim") != env.action_dim or data.get("control", "joint") != env.args.control:
        raise ValueError("Actor initialization observation/action/control mismatch")
    if data.get("usd_sha256") != hashlib.sha256(env.args.usd.read_bytes()).hexdigest():
        raise ValueError("Actor initialization USD differs")
    policy = {key: value for key, value in data["model"].items() if key.startswith("actor.") or key == "log_std"}
    result = model.load_state_dict(policy, strict=False)
    if result.unexpected_keys or any(not key.startswith("critic.") for key in result.missing_keys):
        raise ValueError("Incomplete actor initialization")
    return {"kind": "ppo_actor_only", "source_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "source_task_version": data["task_version"], "source_iteration": data["iteration"],
            "critic_optimizer_rng_curriculum_restored": False}


def evaluate(env, model, args, checkpoint_iteration):
    # Default tests maximum starting difficulty. Fixed goal geometry is preserved;
    # only the legacy goal mode also increases the transport distance here.
    if args.eval_task == "full":
        env.task_stage, env.curriculum = "full", 1.0
        if args.task != "full":
            env.prepare_task()
    obs = env.reset()
    initial = {"cube_position": env.cube.get_transforms()[:, :3].cpu().tolist(),
               "cube_orientation_xyzw": env.cube.get_transforms()[:, 3:].cpu().tolist(),
               "joint_position": env.robot.get_dof_positions().cpu().tolist(),
               "grip_position": env.grip_pose()[0].cpu().tolist(), "goal": env.goal.cpu().tolist()}
    verification_steps = getattr(args, "verify_steps", 0)
    if verification_steps and env.task_stage != "full":
        raise ValueError("Post-completion verification requires a full-task stage, not the lift subtask")
    audit = CompletionAudit(env.num_envs, env.device, verification_steps)
    progress = ProgressAudit(env.num_envs, env.device)
    active = audit.active
    results = []
    trace = []
    start = time.perf_counter()
    model.eval()
    recorder = None
    recording_index = getattr(args, "record_env", 0)
    if args.record:
        from .recording import Recorder
        recorder = Recorder(env, args.record)
    try:
        with torch.no_grad():
            for step in range(env.max_episode_steps + verification_steps):
                raw, _, _ = model.act(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(raw, reset_done=False)
                active_before = active.clone()
                progress.update(info, info["finger"], active_before, step + 1)
                finished = audit.update(info["objective_success"], info["failure"], term, trunc)
                if bool(active_before[0]) and (step % 15 == 0 or bool(finished[0])):
                    trace.append({"step": step + 1, "action": raw[0].tanh().cpu().tolist(),
                                  "grip": env.grip_pose()[0][0].cpu().tolist(),
                                  "cube": info["cube_pos"][0].cpu().tolist(),
                                  "cube_velocity": env.cube.get_velocities()[0].cpu().tolist(),
                                  "finger": env.robot.get_dof_positions()[0, env.finger].item(),
                                  "lift_stable_steps": env.lift_stable_steps[0].item()})
                if recorder and bool(active_before[recording_index]):
                    mode = "PPO joint" if args.control == "joint" else "PPO XYZ+grip / IK servo"
                    if env.task_stage == "lift":
                        mode = "PPO LIFT / IK servo" if args.control == "cartesian" else "PPO LIFT / joint"
                    outcome = f"lift_ok={bool(info['objective_success'][recording_index])}" if env.task_stage == "lift" else f"full_ok={bool(info['success'][recording_index])}"
                    if verification_steps:
                        outcome += f" verified={bool(audit.passed[recording_index])}"
                    recorder.capture(f"{mode} d={env.curriculum:.2f} | env {recording_index} step {step + 1} | {outcome}")
                for index in finished.nonzero().flatten().cpu().tolist():
                    results.append({"environment": index,
                                    "goal_case": case_name(args.goal_request, index) if getattr(args, "goal_mode", "curriculum") == "multi_goal" else None,
                                    "success": bool(audit.initial_success[index]) if env.task_stage == "full" else bool(info["success"][index]),
                                    "objective_success": bool(audit.initial_success[index]),
                                    "verified_success": bool(audit.passed[index]) if verification_steps else None,
                                    "lifted": bool(info["ever_lifted"][index]),
                                    "goal_distance": float(info["goal_distance"][index]),
                                    "grip_distance": float(info["grip_distance"][index]),
                                    "finger": float(info["finger"][index]),
                                    "stable_steps": int(info["stable_steps"][index]),
                                    "full_task_checks_false_at_end": [name for name in ("ever_lifted", "at_goal", "quiet", "released")
                                                                       if not bool(info[name][index])],
                                    "cube_position": info["cube_pos"][index].cpu().tolist(),
                                    "milestone_first_steps": progress.episode(index),
                                    "steps": int(info["episode_length"][index])})
                if not active.any() or time.perf_counter() - start > args.max_seconds:
                    break
    finally:
        if recorder:
            recorder.close()
    report = {"task_version": task_version(env), "control": args.control, "checkpoint": str(args.checkpoint),
              "goal_mode": getattr(args, "goal_mode", "curriculum"),
              "goal_contract": contract(getattr(args, "goal_mode", "curriculum")),
               "goal_request": getattr(args, "goal_request", None),
               "training_goal_curriculum": goal_curriculum_state(env),
               "training_goal_curriculum_applied": False,
               "evaluation_goal_scope": (getattr(args, "goal_request", None) or {}).get("kind", "legacy"),
              "gravity_compensation": args.gravity_compensation,
              "gripper_damping": args.gripper_damping,
              "gripper_max_velocity": args.gripper_max_velocity,
              "solver_velocity_iterations": args.solver_velocity_iterations,
              "physics_hz": args.physics_hz,
              "success_bonus": getattr(args, "success_bonus", 100.),
              "reward_scale": getattr(args, "reward_scale", 1.),
              "completion_hold_steps": getattr(args, "completion_hold_steps", 10),
              "completion_retry_mode": getattr(args, "completion_retry_mode", "retry"),
              "control_ablation": getattr(env, "control_ablation", {}),
              "checkpoint_iteration": checkpoint_iteration,
              "checkpoint_sha256": env.loaded_checkpoint_sha256, "seed": args.seed,
              "recorded_environment": recording_index if args.record else None,
              "recorded_environment_prim": recorder.environment_path if recorder else None,
              "evaluation_task": args.eval_task, "stage": env.task_stage, "curriculum": env.curriculum,
              "requested_episodes": env.num_envs, "completed_episodes": len(results),
              "incomplete_episodes": int(active.sum().item()), "verification_steps": verification_steps,
              "verified_successes": sum(bool(r["verified_success"]) for r in results) if verification_steps else None,
              "objective_successes": sum(r["objective_success"] for r in results),
              "successes": sum(r["success"] for r in results), "lifted_episodes": sum(r["lifted"] for r in results),
              "milestone_counts": progress.counts(),
              "seconds": time.perf_counter() - start, "initial_conditions": initial,
              "trace_environment0": trace, "episodes": results}
    (args.run_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PPO_EVAL " + json.dumps({k: v for k, v in report.items() if k not in ("episodes", "initial_conditions", "trace_environment0")}), flush=True)


def run(env, args):
    cfg = PPOConfig(learning_rate=getattr(args, "learning_rate", 3e-4))
    env.prepare_task()
    model = ActorCritic(env.obs_dim, env.action_dim).to(env.device)
    learner = PPO(model, cfg)
    if args.initialize_from:
        env.initialization = initialize_actor(args.initialize_from, model, env)
        (args.run_dir / "initialization.json").write_text(json.dumps(env.initialization, indent=2), encoding="utf-8")
    initial_iteration = load_checkpoint(args.checkpoint, model, learner, env, args.mode == "train") if args.checkpoint else 0
    if args.mode == "eval":
        evaluate(env, model, args, initial_iteration)
        return
    obs = env.reset()
    history = deque(maxlen=1024)
    steps_at_level = 0
    episode_count = 0
    cumulative_successes = 0
    cumulative_lifts = 0
    cumulative_objectives = 0
    start = time.perf_counter()
    last_iteration = initial_iteration
    checkpoint = args.run_dir / "latest.pt"
    stop_reason = "iteration_limit"
    with (args.run_dir / "metrics.jsonl").open("x", encoding="utf-8") as log:
        for iteration in range(initial_iteration + 1, initial_iteration + args.iterations + 1):
            buffers = {key: [] for key in ("obs", "actions", "logp", "values", "next_values", "rewards", "term", "trunc")}
            with torch.no_grad():
                for _ in range(cfg.horizon):
                    raw, logp, values = model.act(obs)
                    next_obs, reward, term, trunc, info = env.step(raw)
                    next_values = model.critic(info["final_observation"]).squeeze(-1)
                    items = (obs, raw, logp, values, next_values, reward, term, trunc)
                    for key, item in zip(buffers, items):
                        buffers[key].append(item)
                    done_ids = (term | trunc).nonzero().flatten()
                    if len(done_ids):
                        fields = (info["success"][done_ids].float(), info["ever_lifted"][done_ids].float(),
                                               info["episode_return"][done_ids], info["goal_distance"][done_ids],
                                               info["objective_success"][done_ids].float())
                        if getattr(env, "goal_curriculum_mode", "none") != "none":
                            fields += (info["goal_bin"][done_ids].float(),)
                        records = torch.stack(fields, -1).cpu().tolist()
                        history.extend(records)
                        episode_count += len(records)
                        cumulative_successes += sum(int(record[0]) for record in records)
                        cumulative_lifts += sum(int(record[1]) for record in records)
                        cumulative_objectives += sum(int(record[4]) for record in records)
                    obs = next_obs
            batch = {key: torch.stack(items) for key, items in buffers.items()}
            adv, ret = generalized_advantage(batch["rewards"], batch["values"], batch["next_values"], batch["term"], batch["trunc"], cfg.gamma, cfg.gae_lambda)
            metrics = learner.update(batch["obs"].flatten(0, 1), batch["actions"].flatten(0, 1), batch["logp"].flatten(), ret.flatten(), adv.flatten())
            elapsed = time.perf_counter() - start
            metrics.update(iteration=iteration, seconds=elapsed, transitions=(iteration - initial_iteration) * cfg.horizon * env.num_envs,
                           transitions_per_second=(iteration - initial_iteration) * cfg.horizon * env.num_envs / elapsed,
                           episodes=episode_count, history_episodes=len(history), curriculum=env.curriculum, stage=env.task_stage,
                           cumulative_successes=cumulative_successes, cumulative_lifts=cumulative_lifts,
                           cumulative_objectives=cumulative_objectives,
                           train_objective_rate=sum(r[4] for r in history) / max(1, len(history)),
                           train_success_rate=sum(r[0] for r in history) / max(1, len(history)),
                           train_lift_rate=sum(r[1] for r in history) / max(1, len(history)),
                           train_return=sum(r[2] for r in history) / max(1, len(history)),
                           train_goal_distance=sum(r[3] for r in history) / max(1, len(history)))
            metrics["goal_curriculum"] = goal_curriculum_state(env)
            steps_at_level += cfg.horizon
            if getattr(env, "goal_curriculum_mode", "none") != "none":
                level, difficulty, evidence = goal_progress(history, steps_at_level, env.goal_curriculum_level, env.curriculum)
                metrics["goal_curriculum_progress"] = evidence
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print("PPO_TRAIN " + json.dumps(metrics), flush=True)
            last_iteration = iteration
            if getattr(env, "goal_curriculum_mode", "none") != "none":
                if (level, difficulty) != (env.goal_curriculum_level, env.curriculum):
                    anchor = args.run_dir / f"completed_goal_{env.goal_curriculum_level:.2f}_start_{env.curriculum:.2f}_iteration{iteration}.pt"
                    save_checkpoint(anchor, model, learner, env, iteration, cfg)
                    change = {"iteration": iteration, "from_goal_level": env.goal_curriculum_level,
                              "goal_level": level, "from_difficulty": env.curriculum,
                              "difficulty": difficulty, "completed_checkpoint": anchor.name, **evidence}
                    env.goal_curriculum_level, env.curriculum = level, difficulty
                    obs = env.prepare_task()
                    history.clear()
                    steps_at_level = 0
                    with (args.run_dir / "curriculum.jsonl").open("a", encoding="utf-8") as changes:
                        changes.write(json.dumps(change) + "\n")
                    print("PPO_GOAL_CURRICULUM " + json.dumps(change), flush=True)
            elif args.task != "full":
                increment = .1 if args.task in ("staged_precision", "staged_whole") else .25
                stage, difficulty = next_stage(env.task_stage, env.curriculum, len(history), metrics["train_objective_rate"], elapsed_steps=steps_at_level, increment=increment)
                if (stage, difficulty) != (env.task_stage, env.curriculum):
                    completed_checkpoint = args.run_dir / f"completed_{env.task_stage}_{env.curriculum:.2f}_iteration{iteration}.pt"
                    save_checkpoint(completed_checkpoint, model, learner, env, iteration, cfg)
                    change = {"iteration": iteration, "from_stage": env.task_stage, "from_difficulty": env.curriculum,
                              "stage": stage, "difficulty": difficulty, "objective_rate": metrics["train_objective_rate"],
                              "completed_checkpoint": completed_checkpoint.name}
                    env.task_stage, env.curriculum = stage, difficulty
                    # Between completed rollouts only. Reinitialize ALL environments
                    # so no unfinished episode is measured under a different task.
                    obs = env.prepare_task()
                    history.clear()
                    steps_at_level = 0
                    with (args.run_dir / "curriculum.jsonl").open("a", encoding="utf-8") as changes:
                        changes.write(json.dumps(change) + "\n")
                    print("PPO_CURRICULUM " + json.dumps(change), flush=True)
            elif len(history) >= 256 and metrics["train_success_rate"] >= .6 and env.curriculum < 1:
                env.curriculum = min(1.0, env.curriculum + .25)
                history.clear()
            if iteration % 10 == 0:
                # A later update can regress. Preserve periodic candidates for
                # independent evaluation; this does not label them as successful.
                save_checkpoint(args.run_dir / f"iteration_{iteration}.pt", model, learner, env, iteration, cfg)
                save_checkpoint(checkpoint, model, learner, env, iteration, cfg)
            if elapsed >= args.max_seconds:
                stop_reason = "time_limit"
                break
            if (args.run_dir / "STOP").exists():
                stop_reason = "stop_file"
                break
        save_checkpoint(checkpoint, model, learner, env, last_iteration, cfg)
    summary = {"task_version": task_version(env), "algorithm": "PPO", "control": args.control, "teacher_samples": 0,
               "goal_mode": getattr(args, "goal_mode", "curriculum"),
               "goal_contract": contract(getattr(args, "goal_mode", "curriculum")),
               "goal_curriculum": goal_curriculum_state(env),
               "task_transfer": getattr(env, "task_transfer", None),
               "gravity_compensation": args.gravity_compensation, "gripper_damping": args.gripper_damping,
               "gripper_max_velocity": args.gripper_max_velocity,
               "solver_velocity_iterations": args.solver_velocity_iterations,
               "physics_hz": args.physics_hz,
               "success_bonus": getattr(args, "success_bonus", 100.),
               "reward_scale": getattr(args, "reward_scale", 1.),
               "completion_hold_steps": getattr(args, "completion_hold_steps", 10),
               "completion_retry_mode": getattr(args, "completion_retry_mode", "retry"),
               "initialization": getattr(env, "initialization", {"kind": "random"}),
               "stage": env.task_stage, "curriculum": env.curriculum, "cumulative_objectives": cumulative_objectives,
               "iteration": last_iteration, "seconds": time.perf_counter() - start,
               "episodes": episode_count, "cumulative_successes": cumulative_successes, "cumulative_lifts": cumulative_lifts,
               "stop_reason": stop_reason, "checkpoint": str(checkpoint),
               "training_metrics_are_not_held_out_evaluation": True}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PPO_SAVED " + json.dumps(summary), flush=True)

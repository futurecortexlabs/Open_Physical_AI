"""GPU-vectorized CRX pick-and-place PPO. Existing DAgger artifacts are not used.

Run with the Python environment that contains Isaac Sim 6 and CUDA PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from ppo.goals import GOAL_MODES
from ppo.multigoal import CASES, SUITE_VERSION, absolute_request, parse_instruction
from ppo.goal_curriculum import MODES as GOAL_CURRICULA


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "mechanics", "train", "eval"), default="probe")
    parser.add_argument("--usd", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "crx20ia_l+Robotiq_2F_85_edit+rsd455.usd")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--control", choices=("joint", "cartesian"), default="joint",
                        help="joint: 6 joints + gripper; cartesian: XYZ + gripper with low-level IK")
    parser.add_argument("--task", choices=("full", "staged", "staged_hold", "staged_precision", "staged_whole"), default="full",
                        help="full: v2; staged: v3; staged_hold: v4; staged_precision: v5; staged_whole: v6 trains a short complete cycle first")
    parser.add_argument("--initial-difficulty", type=float, default=0.,
                        help="new train run only: initial curriculum difficulty (0..1), recorded in run arguments")
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="curriculum",
                        help="right_10cm: fixed 10 cm floor displacement to the overview camera's right from the initial cube")
    parser.add_argument("--retarget", action="store_true",
                        help="train with checkpoint only: explicitly change goal mode while restoring PPO/Adam/RNG")
    parser.add_argument("--goal-curriculum", choices=GOAL_CURRICULA, default="none",
                        help="train only: progressively expand destinations; evaluation always uses full goal scope")
    goal_input = parser.add_mutually_exclusive_group()
    goal_input.add_argument("--instruction", help="multi_goal eval: e.g. 右20cm, 左10cm, 右15cm 奥10cm")
    goal_input.add_argument("--goal-xy", type=float, nargs=2, metavar=("X_M", "Y_M"),
                            help="multi_goal eval: absolute robot-local floor XY in meters")
    goal_input.add_argument("--goal-suite", action="store_true", help="multi_goal eval: balanced 16-command benchmark")
    parser.add_argument("--eval-task", choices=("full", "trained"), default="full",
                        help="eval only: full tests maximum start-height/noise difficulty; trained uses saved difficulty; both preserve goal mode")
    parser.add_argument("--verify-steps", type=int, default=0,
                        help="full-stage eval: continue the same policy after success and require this many additional stable steps")
    parser.add_argument("--gravity-compensation", action="store_true",
                        help="apply model-based gravity feed-forward to J1..J6; gravity on objects stays enabled")
    parser.add_argument("--gripper-damping", type=float,
                        help="optional runtime override of finger_joint drive damping; source USD is not edited")
    parser.add_argument("--gripper-max-velocity", type=float,
                        help="optional finger_joint speed limit in rad/s; source USD is not edited")
    parser.add_argument("--solver-velocity-iterations", type=int,
                        help="optional runtime articulation/cube velocity solver iterations (1..255)")
    parser.add_argument("--physics-hz", type=int, choices=(60, 120, 240), default=60,
                        help="physics integration frequency; policy frequency remains 30 Hz")
    parser.add_argument("--eval-control-ablation", action="store_true",
                        help="eval only: explicitly allow changed servo/solver settings; record all differences")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="PPO optimizer learning rate; resume requires the checkpoint's setting")
    parser.add_argument("--success-bonus", type=float, default=100.,
                        help="terminal objective bonus; does not change success criteria")
    parser.add_argument("--reward-scale", type=float, default=1.,
                        help="uniform scale applied to all rewards; recorded in checkpoints")
    parser.add_argument("--completion-hold-steps", type=int, default=10,
                        help="training full-task objective: continuous stable placement steps (10..450); 70 includes 2 seconds after initial success")
    parser.add_argument("--completion-retry-mode", choices=("retry", "fail"), default="retry",
                        help="training: fail ends an episode if it loses stability after first success, matching strict verification")
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initialize-from", type=Path,
                        help="train only: reuse PPO actor weights, but start a fresh critic, Adam, RNG and curriculum")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--record-env", type=int, default=0,
                        help="eval: which environment in the evaluated batch to record")
    parser.add_argument("--profile", action="store_true", help="profile probe mode only")
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.cpu_threads < 1:
        parser.error("num-envs, iterations and max-seconds must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if any(not math.isfinite(value) or value <= 0 for value in (args.success_bonus, args.reward_scale)):
        parser.error("--success-bonus and --reward-scale must be finite and positive")
    if not 10 <= args.completion_hold_steps <= 450:
        parser.error("--completion-hold-steps must be in 10..450")
    if not 0 <= args.initial_difficulty <= 1 or (args.initial_difficulty and (args.mode != "train" or args.checkpoint)):
        parser.error("--initial-difficulty must be in 0..1 and requires a new train run without --checkpoint")
    if args.goal_mode != "curriculum" and args.task != "staged_whole":
        parser.error("fixed goal mode requires --task staged_whole")
    if args.retarget and (args.mode != "train" or not args.checkpoint):
        parser.error("--retarget requires --mode train and --checkpoint")
    if args.goal_curriculum != "none" and (args.mode != "train" or args.goal_mode != "multi_goal"):
        parser.error("--goal-curriculum requires train with multi_goal; never narrows evaluation")
    supplied_goal = args.instruction is not None or args.goal_xy is not None or args.goal_suite
    if supplied_goal and (args.mode != "eval" or args.goal_mode != "multi_goal"):
        parser.error("explicit destinations require --mode eval --goal-mode multi_goal; training samples random goals")
    if args.goal_suite and args.num_envs % len(CASES):
        parser.error("--goal-suite requires num-envs divisible by 16")
    args.goal_request = None
    if args.goal_mode == "multi_goal":
        try:
            args.goal_request = (parse_instruction(args.instruction) if args.instruction is not None else
                                 absolute_request(*args.goal_xy) if args.goal_xy is not None else
                                 {"kind": "suite", "version": SUITE_VERSION} if args.goal_suite else {"kind": "random"})
        except ValueError as error:
            parser.error(str(error))
    if not args.usd.is_file():
        parser.error(f"USD not found: {args.usd}")
    if args.mode == "eval" and not args.checkpoint:
        parser.error("eval requires --checkpoint")
    if args.record and (args.mode not in ("eval", "mechanics") or (args.mode == "mechanics" and args.num_envs != 1)):
        parser.error("record requires eval, or mechanics with num-envs 1")
    if not 0 <= args.record_env < args.num_envs or (args.record_env and not args.record):
        parser.error("--record-env must be inside the recorded batch")
    if args.record and args.record.exists():
        parser.error(f"Refusing to overwrite video: {args.record}")
    if args.profile and args.mode != "probe":
        parser.error("--profile is supported only in probe mode")
    if args.mode == "mechanics" and args.control != "joint":
        parser.error("The scripted mechanics diagnostic requires --control joint")
    if args.mode == "mechanics" and args.task != "full":
        parser.error("The scripted mechanics diagnostic requires --task full")
    if args.eval_task != "full" and args.mode != "eval":
        parser.error("--eval-task trained requires --mode eval")
    if args.verify_steps < 0 or (args.verify_steps and args.mode != "eval"):
        parser.error("--verify-steps must be nonnegative and requires eval of a full-task stage")
    if args.eval_control_ablation and args.mode != "eval":
        parser.error("--eval-control-ablation requires --mode eval")
    if args.solver_velocity_iterations is not None and not 1 <= args.solver_velocity_iterations <= 255:
        parser.error("--solver-velocity-iterations must be in 1..255")
    if args.initialize_from and (args.mode != "train" or args.checkpoint):
        parser.error("--initialize-from requires train and cannot be combined with --checkpoint")
    if args.gripper_damping is not None:
        if not math.isfinite(args.gripper_damping) or args.gripper_damping < 0:
            parser.error("--gripper-damping must be finite and nonnegative")
    if args.gripper_max_velocity is not None:
        if not math.isfinite(args.gripper_max_velocity) or args.gripper_max_velocity <= 0:
            parser.error("--gripper-max-velocity must be finite and positive")
    for checkpoint in (args.checkpoint, args.initialize_from):
        if checkpoint and not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")
    return args


def main():
    args = parse_args()
    args.run_dir = args.run_dir or Path(__file__).resolve().parent / "runs" / datetime.now().strftime("ppo_%Y%m%d_%H%M%S_%f")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "arguments.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    source_dir = args.run_dir / "source"
    source_dir.mkdir()
    entry = Path(__file__).resolve()
    sources = [entry, *sorted((entry.parent / "ppo").glob("*.py"))]
    hashes = {}
    for source in sources:
        relative = source.relative_to(entry.parent)
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (args.run_dir / "source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    from isaacsim import SimulationApp

    config = {"headless": not args.gui, "multi_gpu": False, "limit_cpu_threads": args.cpu_threads}
    if not args.gui and args.record is None:
        config.update(renderer="MinimalRendering", disable_viewport_updates=True)
    app = SimulationApp(config)
    exit_code = 0
    physics_guard = None
    try:
        import torch
        from ppo.environment import PickPlaceEnvironment
        from ppo.physics_guard import PhysicsLogGuard

        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        import sys
        runtime = {"python": sys.version, "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                   "physics_device": args.device, "cpu_threads": args.cpu_threads,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        (args.run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
        physics_guard = PhysicsLogGuard(args.run_dir)
        env = PickPlaceEnvironment(app, args, physics_guard)
        if args.mode == "probe":
            if args.profile:
                import cProfile
                import pstats
                profiler = cProfile.Profile()
                result = profiler.runcall(env.probe)
                profiler.dump_stats(str(args.run_dir / "profile.pstats"))
                with (args.run_dir / "profile.txt").open("w", encoding="utf-8") as profile_log:
                    pstats.Stats(profiler, stream=profile_log).sort_stats("cumulative").print_stats(40)
            else:
                result = env.probe()
            (args.run_dir / "probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print("PPO_PROBE " + json.dumps(result), flush=True)
        elif args.mode == "mechanics":
            from ppo.diagnostics import mechanical_test
            mechanical_test(env, args)
        else:
            from ppo.training import run
            run(env, args)
    except BaseException:
        exit_code = 1
        import traceback
        failure = traceback.format_exc()
        print(failure, flush=True)
        (args.run_dir / "failure.txt").write_text(failure, encoding="utf-8")
    finally:
        try:
            if physics_guard is not None:
                physics_guard.close()
        finally:
            app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()

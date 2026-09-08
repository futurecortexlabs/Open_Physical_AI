"""Bounded PPO-only continuation with maximum-start-difficulty validation per chunk.

This supervises training processes, not robot actions. It never supplies labels,
changes rewards, relaxes task success, or substitutes a scripted controller.
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from export_ppo_report import export_report
from ppo.goals import goal_label
from ppo.multigoal import CASES, SUITE_VERSION, contract


CONTROL_KEYS = ("gripper_damping", "gripper_max_velocity", "solver_velocity_iterations",
                "physics_hz", "success_bonus", "reward_scale", "completion_hold_steps",
                "completion_retry_mode")


def training_command(data, checkpoint, usd, run_dir, num_envs, seconds):
    if (data.get("algorithm") != "PPO" or data.get("teacher_samples") != 0
            or data.get("task_version") != "crx_whole_pick_place_v6"
            or data.get("stage") != "full" or data.get("completion_hold_steps") != 70
            or data.get("completion_retry_mode") != "fail"):
        raise ValueError("Requires a PPO-only whole-task checkpoint with strict 70-step completion")
    command = [sys.executable, "-u", str(Path(__file__).with_name("ppo_pick_place.py")),
               "--mode", "train", "--task", "staged_whole", "--control", data["control"],
               "--goal-mode", data.get("goal_mode", "curriculum"),
               "--checkpoint", str(checkpoint), "--usd", str(usd), "--run-dir", str(run_dir),
               "--num-envs", str(num_envs), "--max-seconds", str(seconds), "--iterations", "100000",
               "--learning-rate", str(data["ppo_config"]["learning_rate"])]
    if data.get("gravity_compensation"):
        command.append("--gravity-compensation")
    if data.get("goal_curriculum"):
        command += ["--goal-curriculum", data["goal_curriculum"]["mode"]]
    for key in CONTROL_KEYS:
        if data.get(key) is not None:
            command += ["--" + key.replace("_", "-"), str(data[key])]
    return command


def fresh_seeds(first_seed, round_index):
    return list(range(first_seed + 4 * round_index, first_seed + 4 * round_index + 4))


def confirmation_seeds(first_seed, round_index, maximum_rounds):
    """Disjoint from every candidate-selection seed and earlier confirmations."""
    return fresh_seeds(first_seed, maximum_rounds + round_index)


def validation_command(checkpoint, usd, evaluation_dir, target_rate, seeds, num_envs):
    return [sys.executable, "-u", str(Path(__file__).with_name("validate_ppo.py")),
            "--checkpoint", str(checkpoint), "--usd", str(usd), "--eval-task", "full",
            "--min-success-rate", str(target_rate), "--seeds", *map(str, seeds),
            "--num-envs", str(num_envs), "--verify-steps", "60", "--max-seconds", "180",
            "--output-dir", str(evaluation_dir)]


def accepted(report, target_rate, seeds, checkpoint_hash, goal_mode="curriculum"):
    """Only a complete maximum-start-difficulty suite with the expected goal passes."""
    expected_episodes = 1024 if goal_mode == "multi_goal" else 256
    if goal_mode == "multi_goal":
        per_goal = report.get("per_goal", [])
        if (report.get("goal_contract") != contract(goal_mode)
                or report.get("goal_request") != {"kind": "suite", "version": SUITE_VERSION}
                or report.get("position_target_met") is not True
                or [row.get("case") for row in per_goal] != [case[0] for case in CASES]
                or any(row.get("episodes") != 64 or row.get("rate", 0) < target_rate
                       or row.get("verified_successes", 0) < math.ceil(64 * target_rate) for row in per_goal)):
            return False
    return (report.get("accepted_in_this_test_scope") is True
            and report.get("evaluation_task") == "full" and report.get("curriculum") == 1.
            and report.get("goal_mode", "curriculum") == goal_mode
            and report.get("min_success_rate") == target_rate and report.get("seeds") == seeds
            and report.get("checkpoint_sha256") == checkpoint_hash
            and report.get("verification_steps", 0) >= 60
            and report.get("requested_episodes") == expected_episodes and report.get("completed_episodes") == expected_episodes
            and not report.get("errors"))


def write_status(output, state):
    temporary = output / "status.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "status.json")


def child(command, log_path, timeout, stop_path, training_dir=None):
    start = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if training_dir is not None and stop_path.exists() and training_dir.exists():
                    # The trainer honors this at a completed PPO update and saves.
                    marker = training_dir / "STOP"
                    if not marker.exists():
                        marker.write_text("Supervisor stop requested. Save at the next update.\n", encoding="utf-8")
                if time.monotonic() - start > timeout:
                    raise TimeoutError("Child process exceeded its bounded runtime; see " + log_path.name)
                time.sleep(1)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--usd", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "crx20ia_l+Robotiq_2F_85_edit+rsd455.usd")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--train-seconds", type=float, default=1800)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--target-rate", type=float, default=.9)
    parser.add_argument("--first-eval-seed", type=int, default=2001)
    parser.add_argument("--confirm-candidate", action="store_true",
                        help="before accepting, require the same frozen policy to pass another fresh 1024-trial multi-goal suite")
    args = parser.parse_args()
    if (args.rounds < 1 or args.num_envs < 1 or not math.isfinite(args.train_seconds)
            or args.train_seconds <= 0 or not math.isfinite(args.target_rate) or not 0 < args.target_rate <= 1):
        parser.error("positive run limits and target-rate in (0, 1] are required")
    import torch
    checkpoint = args.checkpoint.resolve()
    usd = args.usd.resolve()
    output = args.output_dir.resolve()
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Check compatibility before starting work or creating an output directory.
    training_command(data, checkpoint, usd, output / "train_01", args.num_envs, args.train_seconds)
    if not usd.is_file() or hashlib.sha256(usd.read_bytes()).hexdigest() != data.get("usd_sha256"):
        parser.error("USD is missing or does not match the checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    goal_mode = data.get("goal_mode", "curriculum")
    eval_envs = 256 if goal_mode == "multi_goal" else 64
    state = {"status": "starting", "target_scope": "maximum start difficulty, full task plus 2-second verification",
             "goal_mode": goal_mode, "goal_description": goal_label(goal_mode),
             "goal_curriculum": data.get("goal_curriculum"),
             "target_rate": args.target_rate, "maximum_rounds": args.rounds,
             "fresh_confirmation_required": args.confirm_candidate,
             "maximum_training_seconds_per_round": args.train_seconds,
             "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             "teacher_samples": 0, "started_at": datetime.now().astimezone().isoformat(), "rounds": []}
    stop_path = output / "STOP"
    try:
        for index in range(args.rounds):
            if stop_path.exists():
                state["status"] = "stopped"
                break
            training_dir = output / f"train_{index + 1:02d}"
            evaluation_dir = output / f"validation_{index + 1:02d}"
            state.update(status="training", active_run=training_dir.name)
            write_status(output, state)
            command = training_command(data, checkpoint, usd, training_dir, args.num_envs, args.train_seconds)
            code = child(command, output / f"train_{index + 1:02d}.log", args.train_seconds + 300, stop_path, training_dir)
            if code:
                raise RuntimeError(f"Training failed with exit code {code}")
            checkpoint = training_dir / "latest.pt"
            data = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state["latest_checkpoint"] = str(checkpoint.relative_to(output))
            state["goal_curriculum"] = data.get("goal_curriculum")
            if stop_path.exists():
                state["status"] = "stopped"
                break
            seeds = fresh_seeds(args.first_eval_seed, index)
            state.update(status="validating", active_run=evaluation_dir.name)
            write_status(output, state)
            command = validation_command(checkpoint, usd, evaluation_dir, args.target_rate, seeds, eval_envs)
            code = child(command, output / f"validation_{index + 1:02d}.log", 1500, stop_path)
            result = json.loads((evaluation_dir / "validation.json").read_text(encoding="utf-8"))
            if code not in (0, 1) or result.get("errors"):
                raise RuntimeError("Validation infrastructure failed; see validation log")
            passed = accepted(result, args.target_rate, seeds, hashlib.sha256(checkpoint.read_bytes()).hexdigest(), goal_mode)
            if bool(code == 0) != passed:
                raise RuntimeError("Validation exit status does not match its evidence")
            export_report([training_dir], [evaluation_dir / f"seed_{seed}" for seed in seeds],
                          output / f"round_{index + 1:02d}_report.json")
            state["rounds"].append({"round": index + 1, "iteration": data["iteration"], "curriculum": data["curriculum"],
                                    "goal_curriculum": data.get("goal_curriculum"),
                                    "validation": str((evaluation_dir / "validation.json").relative_to(output)),
                                    "verified_successes": result["verified_successes"], "episodes": 4 * eval_envs,
                                    "accepted": passed})
            accepted_model = evaluation_dir / "validated_model.pt"
            if passed and args.confirm_candidate:
                state["rounds"][-1].update(candidate_passed=True, accepted=False)
                if stop_path.exists():
                    state["status"] = "stopped"
                    break
                confirmation_dir = output / f"confirmation_{index + 1:02d}"
                confirm_seeds = confirmation_seeds(args.first_eval_seed, index, args.rounds)
                state.update(status="confirming", active_run=confirmation_dir.name)
                write_status(output, state)
                command = validation_command(accepted_model, usd, confirmation_dir, args.target_rate, confirm_seeds, eval_envs)
                code = child(command, output / f"confirmation_{index + 1:02d}.log", 1500, stop_path)
                confirmation = json.loads((confirmation_dir / "validation.json").read_text(encoding="utf-8"))
                if code not in (0, 1) or confirmation.get("errors"):
                    raise RuntimeError("Confirmation infrastructure failed; see confirmation log")
                passed = accepted(confirmation, args.target_rate, confirm_seeds,
                                  hashlib.sha256(checkpoint.read_bytes()).hexdigest(), goal_mode)
                if bool(code == 0) != passed:
                    raise RuntimeError("Confirmation exit status does not match its evidence")
                export_report([], [confirmation_dir / f"seed_{seed}" for seed in confirm_seeds],
                              output / f"confirmation_{index + 1:02d}_report.json")
                state["rounds"][-1].update(accepted=passed, confirmation=str((confirmation_dir / "validation.json").relative_to(output)),
                                          confirmation_verified_successes=confirmation["verified_successes"])
                accepted_model = confirmation_dir / "validated_model.pt"
            if passed:
                state.update(status="target_met_in_test_scope",
                             accepted_checkpoint=str(accepted_model.relative_to(output)))
                break
        else:
            state["status"] = "budget_exhausted_target_not_met"
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["updated_at"] = datetime.now().astimezone().isoformat()
        write_status(output, state)
        print("PPO_CONTINUATION " + json.dumps(state, ensure_ascii=False), flush=True)
    return 0 if state["status"] == "target_met_in_test_scope" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a fixed full-task validation suite; no training or policy modification."""

import argparse
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

from export_ppo_report import public_data
from summarize_ppo_stages import markdown, summarize_stages
from ppo.goals import GOAL_MODES
from ppo.multigoal import CASES, SUITE_VERSION, contract


def summarize(reports, seeds, num_envs, checkpoint_hash, verification_steps,
              min_success_rate=1., evaluation_task="full", curriculum=1., goal_mode="curriculum"):
    """Do not accept partial suites, missing verification or a different model."""
    valid = (bool(seeds) and len(set(seeds)) == len(seeds) and num_envs > 0
             and len(reports) == len(seeds) and verification_steps >= 60
             and math.isfinite(min_success_rate) and 0 < min_success_rate <= 1
             and evaluation_task in ("full", "trained") and 0 <= curriculum <= 1
             and (evaluation_task != "full" or curriculum == 1.) and goal_mode in GOAL_MODES)
    required = math.ceil(num_envs * min_success_rate) if valid else num_envs + 1
    for seed, report in zip(seeds, reports):
        verified = report.get("verified_successes")
        initial = report.get("successes")
        valid &= (report.get("seed") == seed and report.get("evaluation_task") == evaluation_task
                  and report.get("stage") == "full" and report.get("curriculum") == curriculum
                  and report.get("goal_mode", "curriculum") == goal_mode
                  and report.get("checkpoint_sha256") == checkpoint_hash
                  and report.get("requested_episodes") == num_envs
                  and report.get("completed_episodes") == num_envs
                  and type(initial) is int and type(verified) is int
                  and required <= verified <= initial <= num_envs
                  and report.get("verification_steps") == verification_steps
                  and report.get("incomplete_episodes", 0) == 0
                  and not report.get("control_ablation"))
    position_evidence = {}
    if goal_mode == "multi_goal":
        try:
            stages = summarize_stages(reports, min_success_rate)
            position_valid = (num_envs % len(CASES) == 0
                              and stages["conditions"]["goal_contract"] == contract(goal_mode)
                              and stages["conditions"]["goal_request"] == {"kind": "suite", "version": SUITE_VERSION}
                              and len(stages["per_goal"]) == len(CASES)
                              and stages["target_met_in_this_evaluation"])
            position_evidence = {"goal_contract": contract(goal_mode),
                                 "goal_request": stages["conditions"]["goal_request"],
                                 "per_goal": stages["per_goal"], "position_target_met": bool(position_valid)}
            valid &= position_valid
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            valid = False
            position_evidence = {"position_target_met": False}
    return {**position_evidence, "accepted_in_this_test_scope": bool(valid), "universal_perfection_proven": False,
            "min_success_rate": min_success_rate, "required_successes_per_seed": required,
            "evaluation_task": evaluation_task, "curriculum": curriculum, "goal_mode": goal_mode,
            "seeds": seeds, "requested_episodes": len(seeds) * num_envs,
            "completed_episodes": sum(r.get("completed_episodes", 0) for r in reports),
            "initial_successes": sum(r.get("successes", 0) for r in reports),
            "verified_successes": sum(r.get("verified_successes") or 0 for r in reports),
            "verification_steps": verification_steps, "checkpoint_sha256": checkpoint_hash,
            "evaluations": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--usd", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "crx20ia_l+Robotiq_2F_85_edit+rsd455.usd")
    parser.add_argument("--seeds", type=int, nargs="+", default=[901, 902, 903, 904])
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--verify-steps", type=int, default=60)
    parser.add_argument("--min-success-rate", type=float, default=1.,
                        help="required verified success fraction in EVERY seed; default remains 100%%")
    parser.add_argument("--eval-task", choices=("full", "trained"), default="full",
                        help="full tests normal distance; trained explicitly tests the saved curriculum")
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 1 or args.verify_steps < 60 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("num-envs/max-seconds must be positive, verify-steps must be >=60")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    if not math.isfinite(args.min_success_rate) or not 0 < args.min_success_rate <= 1:
        parser.error("min-success-rate must be finite and in (0, 1]")
    import torch
    checkpoint_bytes = args.checkpoint.read_bytes()
    data = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    versions = {"crx_full_pick_place_v2": "full", "crx_staged_pick_place_v3": "staged",
                "crx_staged_pick_place_v4": "staged_hold", "crx_staged_pick_place_v5": "staged_precision",
                "crx_whole_pick_place_v6": "staged_whole"}
    if data.get("algorithm") != "PPO" or data.get("task_version") not in versions:
        parser.error("unsupported PPO checkpoint")
    if args.eval_task == "trained" and data.get("stage") != "full":
        parser.error("trained validation requires a full-stage checkpoint, not a lift-only task")
    curriculum = data.get("curriculum", 0.) if args.eval_task == "trained" else 1.
    goal_mode = data.get("goal_mode", "curriculum")
    if goal_mode not in GOAL_MODES:
        parser.error("unsupported checkpoint goal mode")
    if goal_mode == "multi_goal" and (data.get("goal_contract") != contract(goal_mode) or args.num_envs % len(CASES)):
        parser.error("multi_goal needs a matching contract and num-envs divisible by 16")
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    output = args.output_dir or Path(__file__).resolve().parent / "runs" / datetime.now().strftime("validation_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    frozen_checkpoint = output / "validated_model.pt"
    with frozen_checkpoint.open("xb") as stream:
        stream.write(checkpoint_bytes)
    reports = []
    errors = []
    for seed in args.seeds:
        run_dir = output / f"seed_{seed}"
        command = [sys.executable, "-u", str(Path(__file__).with_name("ppo_pick_place.py")),
                   "--mode", "eval", "--task", versions[data["task_version"]], "--eval-task", args.eval_task,
                   "--goal-mode", goal_mode,
                   "--control", data.get("control", "joint"), "--checkpoint", str(frozen_checkpoint.resolve()),
                   "--usd", str(args.usd.resolve()), "--num-envs", str(args.num_envs), "--seed", str(seed),
                   "--verify-steps", str(args.verify_steps), "--max-seconds", str(args.max_seconds),
                   "--run-dir", str(run_dir.resolve())]
        if data.get("gravity_compensation"):
            command.append("--gravity-compensation")
        if goal_mode == "multi_goal":
            command.append("--goal-suite")
        for key in ("gripper_damping", "gripper_max_velocity", "solver_velocity_iterations", "physics_hz",
                    "success_bonus", "reward_scale", "completion_hold_steps", "completion_retry_mode"):
            if data.get(key) is not None:
                command.extend(["--" + key.replace("_", "-"), str(data[key])])
        try:
            with (output / f"seed_{seed}.log").open("x", encoding="utf-8") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.max_seconds + 180)
            if process.returncode != 0:
                raise RuntimeError(f"evaluation process exited {process.returncode}")
            report = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
            reports.append(report)
            print("PPO_VALIDATION " + json.dumps({key: report[key] for key in
                  ("seed", "completed_episodes", "successes", "verified_successes")}), flush=True)
            if args.fail_fast and (report.get("verified_successes") or 0) < math.ceil(args.num_envs * args.min_success_rate):
                break
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            errors.append({"seed": seed, "error": str(error)})
            break
    summary = summarize(reports, args.seeds, args.num_envs, checkpoint_hash, args.verify_steps,
                        args.min_success_rate, args.eval_task, curriculum, goal_mode)
    if len(reports) == len(args.seeds) and not errors:
        try:
            stages = summarize_stages(reports, args.min_success_rate)
            summary["accepted_in_this_test_scope"] &= stages["target_met_in_this_evaluation"]
            (output / "stage_rates.json").write_text(json.dumps(stages, ensure_ascii=False, indent=2), encoding="utf-8")
            (output / "stage_rates.md").write_text(markdown(stages), encoding="utf-8")
        except (KeyError, TypeError, ValueError) as error:
            errors.append({"error": "stage evidence validation: " + str(error)})
    summary["errors"] = errors
    summary["accepted_in_this_test_scope"] &= not errors
    (output / "validation.json").write_text(json.dumps(public_data(summary), indent=2), encoding="utf-8")
    compact = {k: v for k, v in summary.items() if k not in ("evaluations", "per_goal")}
    if "per_goal" in summary:
        compact["per_goal"] = [{k: row[k] for k in ("case", "episodes", "verified_successes", "rate")}
                               for row in summary["per_goal"]]
    print("PPO_VALIDATION_SUMMARY " + json.dumps(compact), flush=True)
    return 0 if summary["accepted_in_this_test_scope"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run one destination command with a frozen goal-conditioned PPO in simulation.

This is evaluation only. Parsing changes the target observation, never the policy
weights, robot actions, rewards or task success criteria.
"""

import argparse
from datetime import datetime
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

from ppo.multigoal import absolute_request, contract, parse_instruction
from train_ppo_until_target import CONTROL_KEYS


def command_for(data, checkpoint, usd, output, request, *, num_envs=1, seed=5101,
                eval_task="full", max_seconds=180, record=None):
    if (data.get("algorithm") != "PPO" or data.get("teacher_samples") != 0
            or data.get("task_version") != "crx_whole_pick_place_v6"
            or data.get("stage") != "full" or data.get("goal_mode") != "multi_goal"
            or data.get("goal_contract") != contract("multi_goal")):
        raise ValueError("指示位置対応のPPO-only multi_goalモデルが必要です。固定目標モデルは使えません")
    if request["kind"] == "relative":
        parts = []
        for value, positive, negative in ((request["right_m"], "右", "左"), (request["away_m"], "奥", "手前")):
            if value:
                # Decimal's fixed notation preserves the float round-trip without
                # rounding the destination or emitting unsupported exponent syntax.
                parts.append(f"{positive if value > 0 else negative}{format(Decimal(str(abs(value))), 'f')}m")
        instruction = " ".join(parts)
        parse_instruction(instruction)
        goal_args = ["--instruction", instruction]
    elif request["kind"] == "absolute":
        absolute_request(request["x"], request["y"])
        goal_args = ["--goal-xy", str(request["x"]), str(request["y"])]
    else:
        raise ValueError("A single relative instruction or XY destination is required")
    result = [sys.executable, "-u", str(Path(__file__).with_name("ppo_pick_place.py")),
              "--mode", "eval", "--task", "staged_whole", "--goal-mode", "multi_goal",
              "--checkpoint", str(checkpoint), "--usd", str(usd), "--run-dir", str(output),
              "--control", data["control"], "--num-envs", str(num_envs), "--seed", str(seed),
              "--eval-task", eval_task, "--verify-steps", "60", "--max-seconds", str(max_seconds), *goal_args]
    if data.get("gravity_compensation"):
        result.append("--gravity-compensation")
    for key in CONTROL_KEYS:
        if data.get(key) is not None:
            result += ["--" + key.replace("_", "-"), str(data[key])]
    if record:
        result += ["--record", str(record), "--record-env", "0"]
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    goal = parser.add_mutually_exclusive_group(required=True)
    goal.add_argument("--instruction", help="右20cm、左10cm、奥15cm、右15cm 奥10cmなど")
    goal.add_argument("--goal-xy", type=float, nargs=2, metavar=("X_M", "Y_M"))
    parser.add_argument("--usd", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "crx20ia_l+Robotiq_2F_85_edit+rsd455.usd")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--eval-task", choices=("full", "trained"), default="full")
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--record", type=Path, help="optional video; records environment 0 chosen before execution")
    parser.add_argument("--dry-run", action="store_true", help="validate metadata and print command without starting Isaac Sim")
    args = parser.parse_args()
    if args.num_envs < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("positive num-envs and max-seconds are required")
    if args.record and args.record.exists():
        parser.error("Refusing to overwrite video")
    try:
        request = parse_instruction(args.instruction) if args.instruction is not None else absolute_request(*args.goal_xy)
        import torch
        model_bytes = args.checkpoint.read_bytes()
        data = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
        if hashlib.sha256(args.usd.read_bytes()).hexdigest() != data.get("usd_sha256"):
            raise ValueError("USD differs from the trained model")
        output = (args.output_dir or Path(__file__).resolve().parent / "runs" / datetime.now().strftime("command_%Y%m%d_%H%M%S_%f")).resolve()
        command = command_for(data, output / "model.pt", args.usd.resolve(), output / "evaluation", request,
                              num_envs=args.num_envs, seed=args.seed, eval_task=args.eval_task,
                              max_seconds=args.max_seconds, record=args.record.resolve() if args.record else None)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    evidence = {"instruction": args.instruction, "goal_request": request,
                "checkpoint_sha256": hashlib.sha256(model_bytes).hexdigest(),
                "evaluation_task": args.eval_task, "command": command, "simulation_only": True}
    if args.dry_run:
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    (output / "model.pt").write_bytes(model_bytes)
    (output / "command.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PPO_COMMAND " + json.dumps({k: v for k, v in evidence.items() if k != "command"}, ensure_ascii=False), flush=True)
    with (output / "simulation.log").open("x", encoding="utf-8") as log:
        try:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.max_seconds + 180)
        except subprocess.TimeoutExpired:
            print("Simulation timed out; inspect " + str(output / "simulation.log"), file=sys.stderr)
            return 2
    if process.returncode:
        print("Simulation failed; inspect " + str(output / "simulation.log"), file=sys.stderr)
        return 2
    report = json.loads((output / "evaluation" / "evaluation.json").read_text(encoding="utf-8"))
    from compare_ppo_evaluations import validate_report
    validate_report(report)
    if report["checkpoint_sha256"] != evidence["checkpoint_sha256"] or report["goal_request"] != request:
        raise ValueError("Executed model or destination differs from the request")
    print("PPO_COMMAND_RESULT " + json.dumps({k: report[k] for k in
          ("completed_episodes", "verified_successes", "goal_request", "curriculum")}, ensure_ascii=False))
    print("Report: " + str(output / "evaluation" / "evaluation.json"))
    return 0 if report["verified_successes"] == args.num_envs else 1


if __name__ == "__main__":
    raise SystemExit(main())

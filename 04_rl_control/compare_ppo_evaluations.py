"""Compare completed PPO evaluations only when conditions and initial states match."""

import argparse
import json
import math
from pathlib import Path
from ppo.goals import GOAL_MODES, right_offset_xy
from ppo.multigoal import validate_geometry


CONDITIONS = (
    "task_version", "control", "evaluation_task", "stage", "curriculum", "seed",
    "requested_episodes", "verification_steps", "gravity_compensation",
    "gripper_damping", "gripper_max_velocity", "solver_velocity_iterations",
    "physics_hz", "success_bonus", "reward_scale", "completion_hold_steps",
    "completion_retry_mode",
)


def validate_report(report):
    goal_mode = report.get("goal_mode", "curriculum")
    if goal_mode not in GOAL_MODES:
        raise ValueError("Unknown evaluation goal mode")
    required = (*CONDITIONS, "initial_conditions", "episodes", "checkpoint_sha256",
                "completed_episodes", "incomplete_episodes", "successes", "verified_successes",
                "lifted_episodes", "control_ablation")
    if any(key not in report for key in required):
        raise ValueError("Missing evaluation fields")
    count = report["requested_episodes"]
    if count <= 0 or report["completed_episodes"] != count or report["incomplete_episodes"] != 0:
        raise ValueError("Incomplete evaluation")
    if goal_mode == "multi_goal":
        validate_geometry(report)
    if goal_mode == "right_10cm":
        initial = report["initial_conditions"]
        cubes, goals = initial.get("cube_position", []), initial.get("goal", [])
        if len(cubes) != count or len(goals) != count:
            raise ValueError("Missing fixed-goal initial geometry")
        dx, dy = right_offset_xy()
        for cube, goal in zip(cubes, goals):
            if len(cube) != 3 or len(goal) != 3 or not all(math.isfinite(v) for v in (*cube, *goal)):
                raise ValueError("Invalid fixed-goal geometry")
            if not all(math.isclose(a, b, rel_tol=0., abs_tol=1e-5)
                       for a, b in zip(goal, (cube[0] + dx, cube[1] + dy, .03))):
                raise ValueError("Goal is not 10 cm to the initial cube's screen right")
    if report["stage"] != "full" or report["verification_steps"] < 60 or report["control_ablation"]:
        raise ValueError("Requires full-task post-success verification without control ablation")
    episodes = report["episodes"]
    if len(episodes) != count or sorted(e["environment"] for e in episodes) != list(range(count)):
        raise ValueError("Missing or duplicate episode identities")
    for episode in episodes:
        if any(type(episode.get(key)) is not bool for key in ("success", "verified_success", "lifted")):
            raise ValueError("Episode outcomes must be explicit booleans")
        if episode["verified_success"] and not episode["success"]:
            raise ValueError("Verified success requires initial success")
    for total, key in (("successes", "success"), ("verified_successes", "verified_success"),
                       ("lifted_episodes", "lifted")):
        if report[total] != sum(e[key] for e in episodes):
            raise ValueError("Outcome totals do not match episode evidence")


def failure_groups(report):
    groups = dict.fromkeys(("no_lift", "lift_without_goal", "goal_without_completion",
                           "lost_post_success_condition", "verified"), 0)
    for episode in report["episodes"]:
        if episode["verified_success"]:
            group = "verified"
        elif episode["success"]:
            group = "lost_post_success_condition"
        elif not episode["lifted"]:
            group = "no_lift"
        elif not episode["milestone_first_steps"].get("at_goal_after_lift"):
            group = "lift_without_goal"
        else:
            group = "goal_without_completion"
        groups[group] += 1
    return groups


def compare_reports(baseline, candidate):
    for report in (baseline, candidate):
        validate_report(report)
    differences = [key for key in CONDITIONS if baseline[key] != candidate[key]]
    if baseline.get("goal_mode", "curriculum") != candidate.get("goal_mode", "curriculum"):
        differences.append("goal_mode")
    for key in ("goal_contract", "goal_request"):
        if baseline.get(key) != candidate.get(key):
            differences.append(key)
    if differences:
        raise ValueError("Evaluation conditions differ: " + ", ".join(differences))
    if baseline["initial_conditions"] != candidate["initial_conditions"]:
        raise ValueError("Initial states differ; this is not a matched trial comparison")
    before = {e["environment"]: e for e in baseline["episodes"]}
    after = {e["environment"]: e for e in candidate["episodes"]}
    transitions = dict.fromkeys(("both_verified", "newly_verified", "lost_verified", "neither_verified"), 0)
    names = {(True, True): "both_verified", (False, True): "newly_verified",
             (True, False): "lost_verified", (False, False): "neither_verified"}
    for index, episode in before.items():
        transitions[names[episode["verified_success"], after[index]["verified_success"]]] += 1
    return {
        "kind": "MATCHED_PPO_EVALUATION_COMPARISON",
        "conditions": {**{key: baseline[key] for key in CONDITIONS},
                       "goal_mode": baseline.get("goal_mode", "curriculum"),
                       "goal_contract": baseline.get("goal_contract"), "goal_request": baseline.get("goal_request")},
        "initial_conditions_exactly_equal": True,
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "counts": {key: {"baseline": baseline[key], "candidate": candidate[key],
                         "delta": candidate[key] - baseline[key]}
                   for key in ("lifted_episodes", "successes", "verified_successes")},
        "failure_groups": {"baseline": failure_groups(baseline), "candidate": failure_groups(candidate)},
        "paired_verification_outcomes": transitions,
        "universal_perfection_proven": False,
        "checkpoint_selection_makes_this_a_regression_test": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_reports(json.loads(args.baseline.read_text(encoding="utf-8")),
                             json.loads(args.candidate.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Aggregate stage attainment for one fixed policy, without mixing task conditions."""

import argparse
import json
import math
from pathlib import Path

from compare_ppo_evaluations import CONDITIONS, validate_report
from ppo.goals import goal_label
from ppo.multigoal import CASES


STAGES = (
    ("reached", "接近", "手先と物体の距離が5 cm未満"),
    ("grasp_proxy", "把持（代理判定）", "距離6 cm未満かつ指関節角0.1 rad超。接触力の確認ではない"),
    ("lifted", "持ち上げ", "物体高さ10 cm超、手先距離9 cm未満、指関節角0.1 rad超の履歴"),
    ("at_goal_after_lift", "搬送・目標到達", "持ち上げ履歴があり、物体の目標位置誤差3.5 cm未満"),
    ("quiet_at_goal", "配置位置で静止", "持ち上げ後に目標位置で並進速度5 cm/s未満、角速度0.5 rad/s未満。接地・接触力の確認ではない"),
    ("opened_at_goal", "開放", "持ち上げ後に目標位置で指関節角0.12 rad未満"),
    ("retreated_at_goal", "退避", "目標位置で開放し、手先と物体の距離9 cm超"),
    ("completed", "全工程・初回成功", "持ち上げ履歴・目標位置・静止・開放・退避が10ステップ連続成立"),
    ("verified", "完了後2秒の維持", "初回成功後も同じPPOで60ステップ以上、連続して成功条件を維持"),
)


def stage_rows(episodes):
    total = len(episodes)
    rows = []
    for key, label, definition in STAGES:
        if key == "verified":
            hits = sum(episode["verified_success"] for episode in episodes)
        else:
            recorded = [key in episode["milestone_first_steps"] for episode in episodes]
            if any(recorded) and not all(recorded):
                raise ValueError("Partially recorded milestone: " + key)
            hits = sum(episode["milestone_first_steps"][key] is not None for episode in episodes) if all(recorded) else None
        rows.append({"key": key, "label": label, "definition": definition,
                     "attained": hits, "denominator": total,
                     "rate": hits / total if hits is not None and total else None})
    return rows


def summarize_stages(reports, target_rate=.90):
    if not reports or not math.isfinite(target_rate) or not 0 < target_rate <= 1:
        raise ValueError("Need completed evaluations and a target rate in (0, 1]")
    for report in reports:
        validate_report(report)
        for episode in report["episodes"]:
            for key, step in episode["milestone_first_steps"].items():
                if step is not None and (type(step) is not int or step < 1
                                         or ("steps" in episode and step > episode["steps"])):
                    raise ValueError("Invalid milestone step: " + key)
            for milestone, outcome in (("completed", "success"), ("lifted", "lifted")):
                if milestone in episode["milestone_first_steps"]:
                    if (episode["milestone_first_steps"][milestone] is not None) != episode[outcome]:
                        raise ValueError("Milestone contradicts episode outcome: " + milestone)
        for key, count in report.get("milestone_counts", {}).items():
            if any(key not in episode["milestone_first_steps"] for episode in report["episodes"]):
                raise ValueError("Milestone count lacks episode evidence: " + key)
            if count != sum(episode["milestone_first_steps"][key] is not None for episode in report["episodes"]):
                raise ValueError("Milestone total does not match episode evidence: " + key)
    reference = reports[0]
    shared = tuple(key for key in CONDITIONS if key != "seed")
    if any(report["checkpoint_sha256"] != reference["checkpoint_sha256"] for report in reports):
        raise ValueError("Do not aggregate different policy checkpoints")
    if any(any(report[key] != reference[key] for key in shared) for report in reports):
        raise ValueError("Do not aggregate different task/physics/evaluation conditions")
    if any(report.get("goal_mode", "curriculum") != reference.get("goal_mode", "curriculum") for report in reports):
        raise ValueError("Do not aggregate different goal modes")
    for key in ("goal_contract", "goal_request"):
        if any(report.get(key) != reference.get(key) for report in reports):
            raise ValueError("Do not aggregate different " + key)
    seeds = [report["seed"] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Repeated seed: reruns must not be counted as independent trials")
    episodes = [episode for report in reports for episode in report["episodes"]]
    total = len(episodes)
    rows = stage_rows(episodes)
    per_goal = []
    if (reference.get("goal_request") or {}).get("kind") == "suite":
        for name, right, away in CASES:
            group = [episode for episode in episodes if episode["goal_case"] == name]
            if not group:
                raise ValueError("Missing goal case: " + name)
            count = sum(episode["verified_success"] for episode in group)
            label = "・".join(part for part in (
                f"{'右' if right > 0 else '左'}{abs(right)*100:g}cm" if right else "",
                f"{'奥' if away > 0 else '手前'}{abs(away)*100:g}cm" if away else "") if part)
            per_goal.append({"case": name, "label": label, "episodes": len(group),
                             "verified_successes": count, "rate": count / len(group),
                             "stages": stage_rows(group)})
    verified = sum(report["verified_successes"] for report in reports)
    per_seed = [{"seed": report["seed"], "episodes": report["completed_episodes"],
                 "verified_successes": report["verified_successes"],
                 "rate": report["verified_successes"] / report["completed_episodes"]}
                for report in reports]
    return {"kind": "PPO_STAGE_ATTAINMENT_REPORT", "checkpoint_sha256": reference["checkpoint_sha256"],
            "conditions": {**{key: reference[key] for key in shared},
                           "goal_mode": reference.get("goal_mode", "curriculum"),
                           "goal_contract": reference.get("goal_contract"),
                           "goal_request": reference.get("goal_request")}, "seeds": seeds,
            "episodes": total, "stages": rows, "verified_successes": verified,
            "verified_rate": verified / total, "per_seed": per_seed, "per_goal": per_goal, "target_rate": target_rate,
            "target_met_in_this_evaluation": verified / total >= target_rate
                and all(item["rate"] >= target_rate for item in per_seed)
                and all(item["rate"] >= target_rate for item in per_goal),
            "target_requires_each_goal_case": bool(per_goal),
            "target_requires_each_seed": True, "population_success_rate_guaranteed": False,
            "stage_rates_use_all_episodes_not_conditional_denominators": True,
            "milestones_are_independent_attainment_not_a_scripted_sequence": True}


def markdown(report):
    lines = ["# PPO 工程別到達率", "",
             f"対象：{report['conditions']['evaluation_task']} / 難易度{report['conditions']['curriculum']}。",
             f"搬送先：{goal_label(report['conditions'].get('goal_mode', 'curriculum'))}。",
             f"評価seed：{', '.join(map(str, report['seeds']))}、合計{report['episodes']}試行。",
             f"モデルSHA-256：`{report['checkpoint_sha256']}`。", "",
             "| 工程 | 到達数 / 全試行数 | 到達率 |", "|---|---:|---:|"]
    for row in report["stages"]:
        count = f"{row['attained']}/{row['denominator']}" if row["attained"] is not None else "未計測"
        rate = f"{100 * row['rate']:.2f}%" if row["rate"] is not None else "—"
        lines.append(f"| {row['label']} | {count} | {rate} |")
    lines += ["", "全行の分母は全試行数です。前工程の成功者だけを分母にしていません。",
              "各行は独立した到達条件であり、必ず表の順番に通過するとは限りません。把持は代理判定です。",
              "旧評価で個別に記録していなかった項目は、ゼロとせず未計測と表示します。", "",
              "## seed別の最終成功", "", "| seed | 継続成功数 / 全試行数 | 成功率 |", "|---|---:|---:|"]
    lines += [f"| {item['seed']} | {item['verified_successes']}/{item['episodes']} | {100 * item['rate']:.2f}% |"
              for item in report["per_seed"]]
    if report.get("per_goal"):
        lines += ["", "## 指示位置別の最終成功", "", "同じ固定モデルで評価。各位置の分母はその位置を指示した全試行数です。", "",
                  "| 指示 | 継続成功数 / 試行数 | 成功率 |", "|---|---:|---:|"]
        lines += [f"| {item['label']} | {item['verified_successes']}/{item['episodes']} | {100 * item['rate']:.2f}% |"
                  for item in report["per_goal"]]
    elif report["conditions"].get("goal_mode") == "multi_goal":
        lines += ["", "指示条件：`" + json.dumps(report["conditions"]["goal_request"], ensure_ascii=False) + "`。",
                  "これは上記条件だけの評価であり、方向・距離別ベンチマークの合格を意味しません。"]
    lines += ["",
              "## 判定条件", ""]
    lines += [f"- {row['label']}：{row['definition']}。" for row in report["stages"]]
    lines += ["", f"目標：全体と各seedで継続成功率{100 * report['target_rate']:.0f}%以上。",
              "位置別ベンチマークでは、さらに各指示位置でも同じ目標率以上が必要です。",
              "この評価範囲の目標：" + ("達成。" if report["target_met_in_this_evaluation"] else "未達。"),
              "有限回の試験結果であり、任意条件での成功率や実機の安全性を保証しません。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--target-rate", type=float, default=.90)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    for path in (args.output, args.markdown):
        if path is not None and path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    report = summarize_stages([json.loads(path.read_text(encoding="utf-8")) for path in args.evaluation], args.target_rate)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if args.markdown:
        with args.markdown.open("x", encoding="utf-8") as stream:
            stream.write(markdown(report))
    print(json.dumps({key: report[key] for key in ("episodes", "verified_successes", "verified_rate", "target_met_in_this_evaluation")}))


if __name__ == "__main__":
    main()

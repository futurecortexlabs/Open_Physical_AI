"""Export selected completed PPO runs as a path-sanitized, read-only source report."""

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath, PurePosixPath


def public_data(value):
    """Remove machine-specific absolute paths while retaining artifact names."""
    if isinstance(value, dict):
        return {key: public_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_data(item) for item in value]
    if isinstance(value, str):
        windows = PureWindowsPath(value)
        if windows.is_absolute():
            return windows.name
        if PurePosixPath(value).is_absolute():
            return PurePosixPath(value).name
    return value


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export_report(training_runs, evaluation_runs, output):
    report = {"kind": "PPO_EXPERIMENT_REPORT", "training_is_not_held_out_evaluation": True,
              "training_runs": [], "evaluations": []}
    for directory in training_runs:
        summary = read_json(directory / "summary.json")
        metrics = (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        curriculum_path = directory / "curriculum.jsonl"
        curriculum = [json.loads(line) for line in curriculum_path.read_text(encoding="utf-8").splitlines()] if curriculum_path.exists() else []
        report["training_runs"].append({"run": directory.name, "summary": summary,
                                        "last_metrics": json.loads(metrics[-1]),
                                        "metrics": [json.loads(line) for line in metrics], "curriculum": curriculum,
                                        "arguments": read_json(directory / "arguments.json"),
                                        "runtime": read_json(directory / "runtime.json"),
                                        "physics_properties": read_json(directory / "physics_properties.json"),
                                        "physics_messages": read_json(directory / "physics_messages.json"),
                                        "source_hashes": read_json(directory / "source_hashes.json"),
                                        "checkpoint_sha256": hashlib.sha256((directory / "latest.pt").read_bytes()).hexdigest()})
    for directory in evaluation_runs:
        report["evaluations"].append({"run": directory.name, "result": read_json(directory / "evaluation.json"),
                                      "source_hashes": read_json(directory / "source_hashes.json"),
                                      "physics_messages": read_json(directory / "physics_messages.json")})
    # Never replace an existing published report, even accidentally.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(public_data(report), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", action="append", type=Path, default=[])
    parser.add_argument("--evaluation-run", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.training_run and not args.evaluation_run:
        parser.error("At least one completed run is required")
    export_report(args.training_run, args.evaluation_run, args.output)
    print(args.output)

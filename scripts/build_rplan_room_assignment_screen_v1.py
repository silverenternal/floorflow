#!/usr/bin/env python3
"""Compare independent and joint room assignments from matched runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = {
    "mask_iou": "mask_partition_matched_geometric_iou",
    "mask_f1": "mask_partition_instance_f1_at_threshold",
    "mask_type": "mask_partition_matched_type_accuracy",
    "mask_count": "mask_partition_count_abs_error",
    "query_iou": "matched_geometric_iou",
    "query_f1": "instance_f1_at_threshold",
    "query_type": "matched_type_accuracy",
    "query_count": "count_abs_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def best_record(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    history = report.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{path} has no training history")
    return max(history, key=lambda row: float(row["selection_score"]))


def metrics(row: dict) -> dict[str, float]:
    values = row["val"]
    output = {
        name: float(values[key])
        for name, key in METRICS.items()
        if key in values
    }
    output["val_loss"] = float(values["loss"])
    output["selection_score"] = float(row["selection_score"])
    output["epoch"] = int(row["epoch"])
    return output


def main() -> None:
    args = parse_args()
    independent = metrics(best_record(args.independent))
    joint = metrics(best_record(args.joint))
    delta = {
        key: float(joint[key]) - float(independent[key])
        for key in independent.keys() & joint.keys()
        if isinstance(joint[key], (int, float))
    }
    report = {
        "schema_version": "rplan-room-assignment-screen-v1",
        "decision": (
            "reject_joint_assignment"
            if delta["selection_score"] <= 0.0
            else "joint_assignment_candidate"
        ),
        "independent": {
            "report": str(args.independent),
            "best": independent,
        },
        "joint": {
            "report": str(args.joint),
            "best": joint,
        },
        "joint_minus_independent": delta,
        "claim_boundary": (
            "Matched 64-train/32-validation fast screen; not a full-data "
            "or external SOTA result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


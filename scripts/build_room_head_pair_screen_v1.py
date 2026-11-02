#!/usr/bin/env python3
"""Compare two matched room-head training reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--schema-version", default="room-head-pair-v1")
    parser.add_argument(
        "--claim-boundary",
        default=(
            "Matched 64-train/32-validation fast screen; not a full-data "
            "or external SOTA result."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def best_record(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    history = report.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{path} has no history")
    row = max(
        history,
        key=lambda value: float(
            value.get("selection_score", value["best_selection_score"])
        ),
    )
    selected = row["blends"][str(row["best_blend"])]
    return {
        "epoch": int(row["epoch"]),
        "selection_score": float(
            row.get("selection_score", row["best_selection_score"])
        ),
        "best_blend": float(row["best_blend"]),
        "room_iou": float(selected["room_macro_iou"]),
        "instance_f1": float(selected["instance_room_f1"]),
        "type_accuracy": float(selected["instance_type_accuracy"]),
        "count_mae": float(selected["instance_count_abs_error"]),
        "support_iou": float(selected["room_support_iou"]),
    }


def main() -> None:
    args = parse_args()
    baseline = best_record(args.baseline)
    candidate = best_record(args.candidate)
    delta = {
        key: candidate[key] - baseline[key]
        for key in baseline
    }
    joint_gain = (
        delta["room_iou"] > 0.0
        and delta["instance_f1"] > 0.0
        and delta["count_mae"] < 0.0
        and delta["type_accuracy"] >= 0.0
    )
    report = {
        "schema_version": args.schema_version,
        "decision": (
            "candidate"
            if joint_gain and delta["selection_score"] > 0.0
            else "reject_candidate"
        ),
        "baseline": {
            "name": args.baseline_name,
            "report": str(args.baseline),
            "best": baseline,
        },
        "candidate": {
            "name": args.candidate_name,
            "report": str(args.candidate),
            "best": candidate,
        },
        "candidate_minus_baseline": delta,
        "joint_quality_gain": joint_gain,
        "claim_boundary": str(args.claim_boundary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


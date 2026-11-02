#!/usr/bin/env python3
"""Compare two paired floorplan completion reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "room_macro_iou",
    "instance_geometric_iou",
    "instance_room_f1",
    "instance_type_accuracy",
    "fpeval_wall_f1",
    "fpeval_opening_f1",
    "fpeval_room_polygon_f1",
    "furniture_iou",
    "room_support_iou",
    "fpeval_validity",
    "fpeval_watertight",
    "fpeval_edit_cost",
    "fpeval_edit_cost_per_gt_wall",
    "edit_cost_reduction",
    "intent_edit_cost",
    "control_room_pixel_preservation",
    "control_room_support_drift",
    "control_wall_preservation",
    "control_opening_preservation",
    "control_furniture_preservation",
    "room_adjacency_f1",
    "door_contact_room_fraction",
    "door_graph_components",
)

LOWER_IS_BETTER = {
    "fpeval_edit_cost",
    "fpeval_edit_cost_per_gt_wall",
    "intent_edit_cost",
    "control_room_support_drift",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-metrics", type=Path, required=True)
    parser.add_argument("--current-per-case", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--baseline-per-case", type=Path, required=True)
    parser.add_argument("--current-label", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current_metrics = json.loads(args.current_metrics.read_text(encoding="utf-8"))
    baseline_metrics = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    current_rows = json.loads(args.current_per_case.read_text(encoding="utf-8"))
    baseline_rows = json.loads(args.baseline_per_case.read_text(encoding="utf-8"))
    if len(current_rows) != len(baseline_rows):
        raise ValueError("paired case counts differ")
    available = [
        key
        for key in METRICS
        if key in current_rows[0] and key in baseline_rows[0]
    ]
    paired = {
        key: paired_delta(
            current_rows,
            baseline_rows,
            key,
            iterations=args.bootstrap,
        )
        for key in available
    }
    report = {
        "status": "complete",
        "current": args.current_label,
        "baseline": args.baseline_label,
        "cases": len(current_rows),
        "current_control_contract": current_metrics.get(
            "control_contract"
        ),
        "baseline_control_contract": baseline_metrics.get(
            "control_contract"
        ),
        "current_metrics": current_metrics["metrics"],
        "baseline_metrics": baseline_metrics["metrics"],
        "paired": paired,
        "primary_checks": {
            key: {
                "preferred_direction": (
                    "lower" if key in LOWER_IS_BETTER else "higher"
                ),
                "mean_delta": float(paired[key]["mean_delta"]),
                "preferred_mean": (
                    float(paired[key]["mean_delta"]) < 0.0
                    if key in LOWER_IS_BETTER
                    else float(paired[key]["mean_delta"]) > 0.0
                ),
                "ci_excludes_zero": (
                    float(paired[key]["ci_high"]) < 0.0
                    if key in LOWER_IS_BETTER
                    else float(paired[key]["ci_low"]) > 0.0
                ),
            }
            for key in (
                "fpeval_edit_cost",
                "edit_cost_reduction",
                "control_room_pixel_preservation",
                "room_adjacency_f1",
                "fpeval_validity",
            )
            if key in paired
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

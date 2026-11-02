#!/usr/bin/env python3
"""Build a paired report comparing serialized and direct completion controls."""

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
    "conditional_watertight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-metrics", type=Path, required=True)
    parser.add_argument("--cached-per-case", type=Path, required=True)
    parser.add_argument("--direct-metrics", type=Path, required=True)
    parser.add_argument("--direct-per-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached_metrics = json.loads(args.cached_metrics.read_text(encoding="utf-8"))
    direct_metrics = json.loads(args.direct_metrics.read_text(encoding="utf-8"))
    cached_rows = json.loads(args.cached_per_case.read_text(encoding="utf-8"))
    direct_rows = json.loads(args.direct_per_case.read_text(encoding="utf-8"))
    if len(cached_rows) != len(direct_rows):
        raise ValueError("cached and direct row counts differ")

    case_metrics = [
        key
        for key in METRICS
        if key in cached_rows[0] and key in direct_rows[0]
    ]
    paired = {
        key: paired_delta(
            cached_rows,
            direct_rows,
            key,
            iterations=args.bootstrap,
        )
        for key in case_metrics
    }
    mean_deltas = {
        key: float(cached_metrics["metrics"][key])
        - float(direct_metrics["metrics"][key])
        for key in METRICS
    }
    hard_keys = (
        "fpeval_validity",
        "fpeval_watertight",
        "conditional_watertight",
    )
    report = {
        "status": "complete",
        "comparison": "serialized float16 controls vs direct target controls",
        "cases": len(cached_rows),
        "cached_control_contract": cached_metrics.get("control_contract"),
        "direct_control_contract": direct_metrics.get("control_contract"),
        "paired_case_metrics": paired,
        "aggregate_only_metrics": sorted(set(METRICS) - set(case_metrics)),
        "mean_delta": mean_deltas,
        "checks": {
            "validity_identical": all(
                cached_metrics["metrics"][key] == direct_metrics["metrics"][key]
                for key in ("fpeval_validity",)
            ),
            "max_abs_primary_delta_lt_0_01": max(
                abs(mean_deltas[key])
                for key in (
                    "room_macro_iou",
                    "instance_geometric_iou",
                    "fpeval_room_polygon_f1",
                )
            )
            < 0.01,
            "hard_intervals_bounded": all(
                abs(float(paired[key]["mean_delta"])) < 0.01
                for key in hard_keys
                if key in paired
            ),
        },
        "interpretation": (
            "The serialized cache is the official control input. Differences "
            "measure float16 serialization and decode sensitivity, not a new model."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

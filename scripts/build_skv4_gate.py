#!/usr/bin/env python3
"""Fail-closed gate for the skv4 cross-dataset result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap(values: list[float], iterations: int = 2000) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(41)
    means = np.asarray(
        [
            data[generator.integers(0, len(data), len(data))].mean()
            for _ in range(iterations)
        ]
    )
    return {
        "mean": float(data.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    rows = payload["rows"]
    eligible = [row for row in rows if row["target_room_pixels"] > 0]
    keys = (
        "room_macro_iou",
        "instance_geometric_iou",
        "instance_room_f1",
        "instance_type_accuracy",
        "fpeval_wall_f1",
        "fpeval_opening_f1",
        "fpeval_validity",
        "conditional_watertight",
    )
    values = {
        "room_macro_iou": [row["room_macro_iou"] for row in eligible],
        "instance_geometric_iou": [
            row["instance_geometric_iou"] for row in eligible
        ],
        "instance_room_f1": [row["instance_room_f1"] for row in eligible],
        "instance_type_accuracy": [
            row["instance_type_accuracy"] for row in eligible
        ],
        "fpeval_wall_f1": [row["fpeval_wall_f1"] for row in rows],
        "fpeval_opening_f1": [row["fpeval_opening_f1"] for row in rows],
        "fpeval_validity": [row["fpeval_validity"] for row in rows],
        "conditional_watertight": [
            row["fpeval_watertight"]
            for row in rows
            if row["target_watertight"] > 0.5
        ],
    }
    requirements = {
        "case_count_296": len(rows) == 296,
        "room_metric_cases_284": len(eligible) == 284,
        "best_of_n_not_used": not payload.get("best_of_n_used", False),
        **{
            key: len(items) > 0 and float(np.mean(items)) >= 0.95
            for key, items in values.items()
        },
    }
    report = {
        "phase": "B3",
        "dataset": "skv4",
        "task_variant": (
            "contour + room program + coarse room/structure layout control"
        ),
        "requirements": requirements,
        "metrics": {
            key: float(np.mean(items))
            for key, items in values.items()
        },
        "bootstrap": {
            key: bootstrap(items)
            for key, items in values.items()
        },
        "excluded_room_metrics": {
            "no_room_gt_cases": len(rows) - len(eligible),
            "reason": "room macro and instance metrics are undefined without GT rooms",
        },
        "reported_separately": {
            "unconditional_watertight": payload["summary"]["fpeval_watertight"],
            "target_watertight_fraction": payload["summary"]["target_watertight"],
        },
    }
    report["passed"] = all(requirements.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

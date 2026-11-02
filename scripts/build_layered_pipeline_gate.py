#!/usr/bin/env python3
"""Fail-closed publication gate for the complete layered pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def bootstrap(values: list[float], iterations: int) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(31)
    means = np.asarray(
        [
            data[generator.integers(0, len(data), len(data))].mean()
            for _ in range(max(1, iterations))
        ]
    )
    return {
        "mean": float(data.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    rows = json.loads(args.per_case.read_text(encoding="utf-8"))
    keys = (
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
        "conditional_watertight",
    )
    values = {
        key: [float(row[key]) for row in rows if key in row]
        for key in keys
    }
    values["conditional_watertight"] = [
        float(row["fpeval_watertight"])
        for row in rows
        if float(row["target_watertight"]) > 0.5
    ]
    requirements = {
        "frozen_case_count": len(rows) == 503,
        "single_deterministic_sample": bool(
            metrics.get("single_deterministic_sample", False)
        ),
        "best_of_n_not_used": not bool(metrics.get("best_of_n_used", True)),
        **{
            key: (
                (len(items) == 503 if key != "conditional_watertight" else len(items) > 0)
                and float(np.mean(items)) >= 0.95
            )
            for key, items in values.items()
        },
    }
    report = {
        "phase": "A5",
        "task_variant": (
            "contour + room program + coarse room/structure/furniture "
            "layout control"
        ),
        "claim_boundary": {
            "main": "layout-conditioned CAD refinement",
            "not_claimed": "contour-only exact interior reconstruction",
            "contour_only_remains_diagnostic": True,
            "human_evaluation": "excluded_by_user_scope",
            "best_of_n": "excluded",
        },
        "requirements": requirements,
        "metrics": {
            key: float(np.mean(items))
            for key, items in values.items()
        },
        "bootstrap": {
            key: bootstrap(items, args.bootstrap)
            for key, items in values.items()
        },
        "reported_without_hiding": {
            "unconditional_watertight": metrics["metrics"]["fpeval_watertight"],
            "target_watertight_fraction": metrics[
                "target_watertight_fraction"
            ],
            "fpeval_room_face_f1": metrics["metrics"]["fpeval_room_face_f1"],
        },
        "passed": all(requirements.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

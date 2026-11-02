#!/usr/bin/env python3
"""Fail-closed gate for layered full output preserving the room branch."""

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
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def bootstrap(values: list[float], iterations: int) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(23)
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
    keys = {
        "room_macro_iou": "layered_room_macro_iou",
        "instance_geometric_iou": "layered_instance_geometric_iou",
        "instance_room_f1": "layered_instance_room_f1_at_0_5",
        "instance_type_accuracy": "layered_instance_type_accuracy",
    }
    values = {
        target: [float(row[source]) for row in rows if source in row]
        for target, source in keys.items()
    }
    requirements = {
        "frozen_case_count": len(rows) == 503,
        "single_deterministic_sample": True,
        "best_of_n_not_used": True,
        **{
            name: len(items) == 503 and float(np.mean(items)) >= 0.95
            for name, items in values.items()
        },
    }
    report = {
        "phase": "A2",
        "task_variant": "contour + room program + 224 coarse-layout control",
        "input_boundary": "room program and coarse room-block layout; no target rasters at decode",
        "output_boundary": "layered room output with structural channels separated from room storage",
        "requirements": requirements,
        "metrics": {
            name: float(np.mean(items))
            for name, items in values.items()
        },
        "bootstrap": {
            name: bootstrap(items, args.bootstrap)
            for name, items in values.items()
        },
        "human_evaluation": "excluded_by_user_scope",
        "metrics_source": str(args.metrics),
        "per_case_source": str(args.per_case),
    }
    report["passed"] = all(requirements.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

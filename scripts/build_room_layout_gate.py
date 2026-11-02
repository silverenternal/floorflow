"""Publication gate for the layout-controlled room-layout subtask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap(values: list[float], iterations: int) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(17)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    rows = json.loads(args.per_case.read_text())
    requirements = {
        "frozen_case_count": len(rows) == 503,
        "no_best_of_n": not bool(metrics.get("best_of_n_used", False)),
        "room_macro_iou": metrics.get("room_macro_iou_mean", 0.0) >= 0.95,
        "instance_geometric_iou": metrics.get("instance_geometric_iou_mean", 0.0)
        >= 0.95,
        "instance_room_f1": metrics.get("instance_room_f1_at_0_5_mean", 0.0)
        >= 0.50,
        "instance_type_accuracy": metrics.get("instance_type_accuracy_mean", 0.0)
        >= 0.95,
    }
    keys = (
        "room_macro_iou",
        "instance_geometric_iou",
        "instance_room_f1_at_0_5",
        "instance_type_accuracy",
    )
    report = {
        "task_variant": "contour_plus_room_program_plus_224_layout_room_only",
        "input_boundary": "room presence, room program and 224x224 room-block layout",
        "output_boundary": "room layout only; wall/opening/furniture excluded",
        "primary_metric": "category-aware Hungarian matched-instance geometric IoU",
        "human_evaluation": "excluded_by_user_scope",
        "requirements": requirements,
        "absolute_bootstrap": {
            key: bootstrap(
                [float(row[key]) for row in rows if key in row],
                args.bootstrap,
            )
            for key in keys
        },
        "metrics_source": str(args.metrics),
        "per_case_source": str(args.per_case),
        "passed": all(requirements.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

"""Fail-closed publication gate for contour + explicit room-program generation."""

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
    parser.add_argument("--fpeval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--task-variant",
        default="contour_plus_room_program_presence",
    )
    parser.add_argument(
        "--program-input",
        default="ground-truth room-class presence vector",
    )
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    rows = json.loads(args.per_case.read_text())
    vector = json.loads(args.fpeval.read_text())["metrics"]
    requirements = {
        "program_input_declared": True,
        "frozen_case_count": len(rows) == 503,
        "no_best_of_n": not bool(metrics.get("best_of_n_used", False)),
        "matched_class_instance_iou": metrics.get(
            "instance_geometric_iou_mean", 0.0
        )
        >= 0.95,
        "instance_room_f1": metrics.get("instance_room_f1_at_0_5_mean", 0.0)
        >= 0.50,
        "instance_type_accuracy": metrics.get("instance_type_accuracy_mean", 0.0)
        >= 0.90,
        "room_macro_iou": metrics.get("room_macro_iou_mean", 0.0) >= 0.20,
        "room_dataset_macro_iou": metrics.get("room_dataset_macro_iou", 0.0)
        >= 0.20,
        "wall_centerline_f1": metrics.get("wall_centerline_f1_mean", 0.0)
        >= 0.70,
        "fpeval_wall_f1_at_0_05": vector["0.05"]["wall_f1"] >= 0.70,
        "fpeval_opening_f1_at_0_05": vector["0.05"]["opening_f1"] >= 0.50,
        "fpeval_room_f1_at_0_05": vector["0.05"]["room_f1"] >= 0.50,
        "fpeval_validity": vector["validity_fraction"] >= 0.90,
        "fpeval_watertight": vector["watertight_fraction"] >= 0.90,
        "room_support_iou": vector["support_iou_all"] >= 0.95,
    }
    keys = (
        "support_instance_geometric_iou",
        "instance_geometric_iou",
        "instance_room_f1_at_0_5",
        "instance_type_accuracy",
        "room_macro_iou",
        "room_dataset_macro_iou",
    )
    absolute = {
        key: bootstrap(
            [float(row[key]) for row in rows if key in row],
            args.bootstrap,
        )
        for key in keys
        if any(key in row for row in rows)
    }
    report = {
        "task_variant": args.task_variant,
        "program_input": args.program_input,
        "primary_metric": "category-aware Hungarian matched-instance geometric IoU",
        "support_instance_diagnostic": metrics.get(
            "support_instance_geometric_iou_mean"
        ),
        "secondary_metrics": [
            "strict visible room macro IoU",
            "instance type accuracy",
            "wall/opening vector metrics",
        ],
        "human_evaluation": "excluded_by_user_scope",
        "requirements": requirements,
        "absolute_bootstrap": absolute,
        "metrics_source": str(args.metrics),
        "per_case_source": str(args.per_case),
        "fpeval_source": str(args.fpeval),
        "passed": all(requirements.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

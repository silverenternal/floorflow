#!/usr/bin/env python3
"""Build a fail-closed publication gate from full-test metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta as paired_bootstrap
from floorflow.inference.sampling_policy import validate_primary_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--baseline-per-case", type=Path, default=None)
    parser.add_argument("--distribution", type=Path, default=None)
    parser.add_argument("--fpeval-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def bootstrap_mean(values: list[float], iterations: int) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(17)
    means = np.asarray(
        [data[rng.integers(0, len(data), len(data))].mean() for _ in range(max(1, iterations))]
    )
    return {
        "mean": float(data.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    per_case = json.loads(args.per_case.read_text(encoding="utf-8"))
    if "metrics" in metrics and isinstance(metrics["metrics"], dict):
        nested = metrics
        metrics = {f"{key}_mean": value for key, value in nested["metrics"].items()}
        metrics.update({key: value for key, value in nested.items() if key not in {"metrics", "rows"}})
        per_case = nested.get("rows", per_case)
    validity_requirements = {
        "full_test_case_count": len(per_case) >= 500,
        "metrics_best_of_n_not_used": not bool(metrics.get("best_of_n_used", False)),
        "outside_footprint_zero": metrics.get("outside_footprint_fraction_mean", 1.0) <= 1.0e-6,
        "exterior_wall_valid": metrics.get("exterior_band_is_known_wall_mean", 0.0) >= 0.99,
        "room_area_coverage": metrics.get("room_area_coverage_mean", 0.0) >= 0.90,
        "opening_f1": metrics.get("opening_f1_mean", 0.0) >= 0.50,
        "opening_wall_alignment": metrics.get("opening_wall_alignment_mean", 0.0) >= 0.85,
        "furniture_room_containment": metrics.get("furniture_room_containment_mean", 0.0) >= 0.90,
        "room_area_distribution_mae": metrics.get("room_area_distribution_mae_mean", 1.0) <= 0.10,
    }
    research_requirements = {
        "room_macro_iou": metrics.get("room_macro_iou_mean", 0.0) >= 0.20,
        "partition_room_macro_iou": metrics.get("partition_room_macro_iou_mean", 0.0) >= 0.20,
        "room_group_macro_iou": metrics.get("room_group_macro_iou_mean", 0.0) >= 0.20,
        "room_dataset_macro_iou": metrics.get("room_dataset_macro_iou", 0.0) >= 0.20,
        "room_dataset_micro_iou": metrics.get("room_dataset_micro_iou", 0.0) >= 0.20,
        "instance_geometric_iou": metrics.get("instance_geometric_iou_mean", 0.0) >= 0.80,
        "instance_room_f1_at_0_5": metrics.get("instance_room_f1_at_0_5_mean", 0.0) >= 0.50,
        "instance_type_accuracy": metrics.get("instance_type_accuracy_mean", 0.0) >= 0.90,
        "channel_furniture_iou": metrics.get("channel_furniture_iou_mean", 0.0) >= 0.10,
        "channel_opening_f1": metrics.get("channel_opening_f1_mean", 0.0) >= 0.50,
        "channel_wall_iou": metrics.get("channel_wall_iou_mean", 0.0) >= 0.20,
        "interior_wall_iou": metrics.get("interior_wall_iou_mean", 0.0) >= 0.10,
        "wall_centerline_f1": metrics.get("wall_centerline_f1_mean", 0.0) >= 0.70,
        "wall_closure": metrics.get("wall_closure_mean", 0.0) >= 0.90,
    }
    vector_audit = None
    if args.fpeval_report is not None:
        vector = json.loads(args.fpeval_report.read_text(encoding="utf-8"))
        vector_metrics = vector["metrics"]
        vector_audit = {
            "source": str(args.fpeval_report),
            "case_count": vector["case_count"],
        }
        research_requirements.update(
            {
                "fpeval_wall_f1_at_0_05": vector_metrics["0.05"]["wall_f1"] >= 0.70,
                "fpeval_room_f1": vector_metrics["0.05"]["room_f1"] >= 0.50,
                "fpeval_opening_f1": vector_metrics["0.05"]["opening_f1"] >= 0.50,
                "fpeval_validity": vector_metrics["validity_fraction"] >= 0.90,
                "fpeval_watertight": vector_metrics["watertight_fraction"] >= 0.90,
                "room_support_iou": vector_metrics["support_iou_all"] >= 0.95,
            }
        )
    distribution_audit = None
    if args.distribution is not None:
        distribution = json.loads(args.distribution.read_text(encoding="utf-8"))
        distribution_valid, sampling_errors = validate_primary_report(distribution)
        distribution_audit = {
            "source": str(args.distribution),
            "valid_primary_sampling": distribution_valid,
            "errors": sampling_errors,
        }
        validity_requirements["primary_sampling_integrity"] = distribution_valid
    paired = {}
    if args.baseline_per_case is not None:
        baseline = json.loads(args.baseline_per_case.read_text(encoding="utf-8"))
        for key in (
            "room_macro_iou",
            "instance_geometric_iou",
            "instance_room_f1_at_0_5",
            "instance_type_accuracy",
            "furniture_iou",
            "opening_f1",
            "interior_wall_iou",
            "opening_wall_alignment",
            "furniture_room_containment",
        ):
            paired[key] = paired_bootstrap(per_case, baseline, key, args.bootstrap)
    absolute = {
        key: bootstrap_mean(
            [float(row[key]) for row in per_case if key in row],
            args.bootstrap,
        )
        for key in (
            "room_macro_iou",
            "instance_geometric_iou",
            "instance_room_f1_at_0_5",
            "instance_type_accuracy",
            "opening_f1",
            "furniture_iou",
            "interior_wall_iou",
            "wall_centerline_f1",
        )
        if any(key in row for row in per_case)
    }
    report = {
        "metrics": str(args.metrics),
        "per_case": str(args.per_case),
        "validity_requirements": validity_requirements,
        "research_requirements": research_requirements,
        "distribution_audit": distribution_audit,
        "vector_audit": vector_audit,
        "paired_bootstrap": paired,
        "absolute_bootstrap": absolute,
        "validity_passed": all(validity_requirements.values()),
        "research_targets_passed": all(research_requirements.values()),
        "passed": all(validity_requirements.values()) and all(research_requirements.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

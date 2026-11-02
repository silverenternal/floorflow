#!/usr/bin/env python3
"""Build the final contour-only publication gate from frozen evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-raw", type=Path, required=True)
    parser.add_argument("--test-fusion", type=Path, required=True)
    parser.add_argument("--test-overlay", type=Path, required=True)
    parser.add_argument("--skv4", type=Path, required=True)
    parser.add_argument("--diversity", type=Path, required=True)
    parser.add_argument("--efficiency", type=Path, required=True)
    parser.add_argument(
        "--readiness",
        type=Path,
        default=Path("outputs/reports/b5_cvpr_readiness.json"),
    )
    parser.add_argument(
        "--contour-ceiling",
        type=Path,
        default=Path("outputs/reports/room_iou_conditional_ceiling.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap(values: list[float], iterations: int = 1000) -> dict[str, float]:
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
    args = parse_args()
    manifest = load(args.manifest)
    raw = load(args.test_raw)
    fusion = load(args.test_fusion)
    overlay = load(args.test_overlay)
    skv4 = load(args.skv4)
    diversity = load(args.diversity)
    efficiency = load(args.efficiency)
    readiness = load(args.readiness)
    ceiling = load(args.contour_ceiling)

    main_metrics = {
        "cubicasa_test503_raw_room_iou": raw["metrics"]["room_macro_iou"],
        "cubicasa_test503_fusion_room_iou": fusion["metrics"]["room_macro_iou"],
        "cubicasa_test503_overlay_room_iou": overlay["room_metrics"]["overlay"][
            "room_macro_iou"
        ],
        "cubicasa_test503_primary_instance_f1": overlay["room_metrics"][
            "overlay"
        ]["instance_room_f1"],
        "cubicasa_test503_overlay_type_accuracy": overlay["room_metrics"][
            "overlay"
        ]["instance_type_accuracy"],
        "cubicasa_test503_opening_f1": overlay["object_metrics"]["overlay"][
            "opening_f1"
        ],
        "cubicasa_test503_furniture_iou": overlay["object_metrics"]["overlay"][
            "furniture_iou"
        ],
        "cubicasa_test503_wall_iou": overlay["object_metrics"]["overlay"][
            "wall_iou"
        ],
        "skv4_room_iou": skv4["summary"]["room_macro_iou"],
        "skv4_instance_f1": skv4["summary"][
            "instance_f1_at_threshold"
        ],
        "skv4_type_accuracy": skv4["summary"]["matched_type_accuracy"],
        "cfm_sample_expected_room_iou": diversity[
            "expected_room_macro_iou"
        ],
    }
    ci = {
        "cubicasa_test503_raw_room_iou": raw["bootstrap"]["room_macro_iou"],
        **{
            f"skv4_{key}": bootstrap(
                [float(row[key]) for row in skv4["rows"]]
            )
            for key in (
                "room_macro_iou",
                "matched_geometric_iou",
                "instance_f1_at_threshold",
                "matched_type_accuracy",
                "matched_class_constrained_iou",
            )
        },
    }
    requirements = {
        "frozen_manifest": manifest.get("status") == "frozen",
        "contour_only": manifest.get("condition_mode") == "contour_only",
        "target_derived_control_disabled": not manifest.get(
            "target_derived_control", True
        ),
        "best_of_n_disabled": not manifest.get("best_of_n_used", True),
        "cubicasa_503_cases": raw["cases"] == 503,
        "skv4_296_cases": skv4["case_count"] == 296,
        "bootstrap_reported": bool(ci),
        "upper_bounds_reported": True,
        **{
            f"{name}_ge_0_95": float(value) >= 0.95
            for name, value in main_metrics.items()
        },
    }
    report = {
        "status": "fail_closed",
        "conditions": {
            "mode": "contour_only",
            "target_derived_coarse_control": False,
            "single_sample": True,
            "best_of_n_used": False,
        },
        "manifest": str(args.manifest),
        "metrics": main_metrics,
        "bootstrap": ci,
        "upper_bounds": {
            "contour_conditional": {
                "top1_iou": ceiling["summary"]["top1_iou"],
                "neighbour_mean_iou": ceiling["summary"][
                    "neighbour_mean_iou"
                ],
                "strict_0_95_identifiable_fraction": ceiling[
                    "strict_0_95_identifiable_fraction"
                ],
            },
            "layout_conditioned": {
                "requires_target_derived_coarse_control": True,
                "cubicasa_room_iou": readiness["layout_conditioned_results"][
                    "cubicasa_metrics"
                ]["room_macro_iou"],
                "skv4_room_iou": readiness["layout_conditioned_results"][
                    "skv4_metrics"
                ]["room_macro_iou"],
            },
            "scope_note": "layout-conditioned results are not the contour-only main result",
        },
        "diversity": {
            "mmd": diversity["mmd"],
            "coverage": diversity["coverage"],
            "density": diversity["density"],
            "within_condition_diversity": diversity["diversity"],
        },
        "efficiency": efficiency,
        "requirements": requirements,
        "blocking_metrics": [
            name
            for name, value in main_metrics.items()
            if float(value) < 0.95
        ],
    }
    report["passed"] = all(requirements.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

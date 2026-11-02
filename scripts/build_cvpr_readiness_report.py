#!/usr/bin/env python3
"""Build an evidence-backed CVPR readiness report for a frozen candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cad-report",
        type=Path,
        default=ROOT / "outputs/reports/final_retrieval_free_style_test503_v1.json",
    )
    parser.add_argument(
        "--hierarchy",
        type=Path,
        default=ROOT / "outputs/reports/metric_hierarchy_test503_v1.json",
    )
    parser.add_argument(
        "--style-report",
        type=Path,
        default=ROOT / "outputs/reports/final_retrieval_free_style_val8_v1.json",
    )
    parser.add_argument(
        "--model-inventory",
        type=Path,
        default=ROOT / "outputs/reports/model_inventory_v3.json",
    )
    parser.add_argument(
        "--identifiability",
        type=Path,
        default=ROOT / "outputs/reports/goal_blocker_contour_identifiability_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/b5_cvpr_readiness_v2.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cad = read(args.cad_report)
    hierarchy = read(args.hierarchy)
    style = read(args.style_report)
    inventory = read(args.model_inventory)
    identifiability = read(args.identifiability)
    metrics = {
        "room_macro_iou": cad["room_metrics"]["overlay"]["room_macro_iou"],
        "instance_geometric_iou": cad["room_metrics"]["overlay"][
            "instance_geometric_iou"
        ],
        "instance_type_accuracy": cad["room_metrics"]["overlay"][
            "instance_type_accuracy"
        ],
        "fpeval_wall_f1": cad["room_metrics"]["overlay"]["fpeval_wall_f1"],
        "fpeval_opening_f1": cad["room_metrics"]["overlay"][
            "fpeval_opening_f1"
        ],
        "fpeval_room_polygon_f1": cad["room_metrics"]["overlay"][
            "fpeval_room_polygon_f1"
        ],
        "unconditional_watertight": cad["room_metrics"]["overlay"][
            "fpeval_watertight"
        ],
        "conditional_watertight": cad["cad_metrics"]["summary"].get(
            "conditional_watertight"
        ),
        "furniture_iou": cad["object_metrics"]["overlay"]["furniture_iou"],
        "opening_iou": cad["object_metrics"]["overlay"]["opening_iou"],
        "wall_iou": cad["object_metrics"]["overlay"]["wall_iou"],
        "style_separation": style["style_separation"],
        "seed_iou_gap": style["seed_iou_gap"],
        "worst_style_room_iou": style["worst_style_room_iou"],
        "balanced_cad": cad["cad_metrics"]["balanced_secondary"]["score"],
    }
    weak_targets = {
        name: {
            "value": metrics[name],
            "target": 0.95,
            "gap": 0.95 - float(metrics[name]),
            "passed": float(metrics[name]) >= 0.95,
        }
        for name in (
            "room_macro_iou",
            "instance_geometric_iou",
            "instance_type_accuracy",
            "fpeval_wall_f1",
            "fpeval_opening_f1",
            "fpeval_room_polygon_f1",
            "furniture_iou",
            "opening_iou",
            "wall_iou",
        )
    }
    report = {
        "status": "research_incomplete",
        "evaluation_contract": cad["condition_mode"],
        "test_cases": cad["cases"],
        "metrics": metrics,
        "strict_095_targets": weak_targets,
        "hard_floors": cad["cad_metrics"]["hard_gate"],
        "metric_hierarchy_dominant_link": hierarchy["dominant_link"],
        "model_inventory": {
            "core_distribution_cfm_m": inventory["core_distribution_cfm_m"],
            "deployed_total_m": inventory["deployed_total_m"],
            "dataset": inventory["dataset"],
        },
        "identifiability": {
            "strict_0_95_fraction": identifiability[
                "identifiability_evidence"
            ]["strict_0_95_identifiable_fraction"],
            "gt_room_fraction_of_footprint_max": identifiability[
                "hard_floor_definition_error"
            ]["test_target_room_fraction_of_footprint"]["max"],
            "conclusion": identifiability["supported_conclusion"],
        },
        "supported_claims": [
            "Contour-conditioned multi-channel CAD interior reconstruction, not boundary detection.",
            "Single-sample target-free schema validity and footprint exclusion.",
            "Joint room, wall, opening, furniture and support layer evaluation.",
            "Dynamic recursion and MiTA graph adaptation with target-free inference.",
        ],
        "unsupported_claims": [
            "Exact per-case contour-only reconstruction at Room IoU >= 0.95.",
            "CVPR acceptance based on the current object and vector metrics.",
        ],
        "cvpr_gap": sorted(
            (
                {
                    "metric": name,
                    "value": row["value"],
                    "gap_to_0_95": row["gap"],
                }
                for name, row in weak_targets.items()
            ),
            key=lambda row: row["gap_to_0_95"],
            reverse=True,
        ),
        "best_of_n_used": False,
        "external_people_used": False,
        "manual_annotation_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Gate the deterministic distribution-overlay branch for publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--separation", type=Path, required=True)
    parser.add_argument("--seed19", type=Path, required=True)
    parser.add_argument("--seed23", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    val = load(args.val)
    test = load(args.test)
    separation = load(args.separation)
    seed19 = load(args.seed19)
    seed23 = load(args.seed23)
    val_aggregate = val["aggregate"]
    test_aggregate = test["aggregate"]
    separation_mean = float(
        separation["summary"]["room_conditioned_disagreement_mean"]
    )
    seed19_iou = float(seed19["room_metrics"]["overlay"]["room_macro_iou"])
    seed23_iou = float(seed23["room_metrics"]["overlay"]["room_macro_iou"])
    seed_gap = abs(seed19_iou - seed23_iou)
    checks = {
        "val_all_style_iou": float(
            val_aggregate["quality_mean"]
        ) >= 0.12,
        "test_all_style_iou": float(
            test_aggregate["quality_mean"]
        ) >= 0.12,
        "test_worst_style_iou": float(
            test_aggregate["quality_worst"]
        ) >= 0.11,
        "test_coverage": float(
            test_aggregate["coverage_worst"]
        ) >= 0.90,
        "no_footprint_violation": float(
            test_aggregate["outside_footprint_pixels_mean"]
        ) == 0.0,
        "room_conditioned_style_separation": separation_mean >= 0.05,
        "seed_consistency": seed_gap <= 0.02,
        "val_above_flow_gate": float(
            val_aggregate["quality_mean"]
        ) >= 0.10,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "claim_role": "style_controllable_constrained_floorplan_room_decoder",
        "selection": {
            "checkpoint": "outputs/train/distribution_style_prompt_r94_v2/best.pt",
            "retrieval_prior_weight": 0.5,
            "retrieval_top_k": 1,
            "best_of_n_used": False,
            "target_derived_control": False,
            "test_selection_used": False,
        },
        "metrics": {
            "val_all_style_room_iou_mean": val_aggregate["quality_mean"],
            "val_all_style_room_iou_worst": val_aggregate["quality_worst"],
            "test_all_style_room_iou_mean": test_aggregate["quality_mean"],
            "test_all_style_room_iou_worst": test_aggregate["quality_worst"],
            "test_coverage_worst": test_aggregate["coverage_worst"],
            "test_outside_footprint_pixels_mean": test_aggregate[
                "outside_footprint_pixels_mean"
            ],
            "room_conditioned_style_disagreement": separation_mean,
            "seed19_room_iou": seed19_iou,
            "seed23_room_iou": seed23_iou,
            "seed_gap": seed_gap,
        },
        "checks": checks,
        "sources": {
            "val": str(args.val),
            "test": str(args.test),
            "separation": str(args.separation),
            "seed19": str(args.seed19),
            "seed23": str(args.seed23),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

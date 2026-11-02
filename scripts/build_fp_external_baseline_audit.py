#!/usr/bin/env python3
"""Audit which external baselines satisfy the FP hard-comparison contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("outputs/reports/external_baseline_comparison_v1.json"),
    )
    parser.add_argument(
        "--availability",
        type=Path,
        default=Path("outputs/reports/external_baseline_availability_v1.json"),
    )
    parser.add_argument(
        "--house-diffusion",
        type=Path,
        default=Path("outputs/reports/fp_housediffusion_oracle_val58_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_external_baseline_audit_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    availability = json.loads(args.availability.read_text(encoding="utf-8"))
    housediffusion = (
        json.loads(args.house_diffusion.read_text(encoding="utf-8"))
        if args.house_diffusion.is_file()
        else None
    )
    housegan = comparison["oracle_program_table"]["houseganpp"]
    maskplan = comparison["native_table"]["maskplan"]
    rows = [
        {
            "method": "House-GAN++",
            "condition_mode": housegan["condition_mode"],
            "target_derived_control": housegan["target_derived_control"],
            "cases": housegan["cases"],
            "hard_metrics": {
                "footprint_coverage_mean": housegan["summary"][
                    "footprint_coverage_mean"
                ],
                "outside_footprint_pixels_mean": housegan["summary"][
                    "outside_footprint_pixels_mean"
                ],
            },
            "eligible_for_main_table": False,
            "reason": (
                "Oracle room-program condition, different input contract, "
                "and non-zero outside-footprint output."
            ),
        },
        {
            "method": "MaskPLAN",
            "condition_mode": maskplan["condition_mode"],
            "target_derived_control": maskplan["target_derived_control"],
            "cases": maskplan["cases"],
            "hard_metrics": {
                "footprint_coverage_mean": maskplan["summary"][
                    "footprint_coverage_mean"
                ],
                "outside_footprint_pixels_mean": maskplan["summary"][
                    "outside_footprint_pixels_mean"
                ],
            },
            "eligible_for_main_table": False,
            "reason": (
                "Native RPLAN benchmark only; split and decoder interface "
                "are not matched to CubiCasa completion."
            ),
        },
        {
            "method": "HouseDiffusion",
            "condition_mode": "oracle_room_program_adapter",
            "target_derived_control": True,
            "cases": housediffusion["cases"] if housediffusion else 58,
            "hard_metrics": (
                {
                    "footprint_coverage_mean": housediffusion["summary"][
                        "footprint_coverage"
                    ],
                    "outside_footprint_pixels_mean": housediffusion["summary"][
                        "outside_footprint_pixels"
                    ],
                    "room_support_iou_mean": housediffusion["summary"][
                        "room_support_iou"
                    ],
                }
                if housediffusion is not None
                else None
            ),
            "eligible_for_main_table": False,
            "reason": (
                "Oracle room-program condition, different input contract, "
                "and no wall/opening/watertight output."
            ),
        },
    ]
    report = {
        "status": "complete",
        "comparison_source": str(args.comparison.resolve()),
        "availability_source": str(args.availability.resolve()),
        "availability_status": availability["status"],
        "rows": rows,
        "matched_external_baseline_count": 0,
        "contextual_baseline_count": (
            3 if housediffusion is not None else 2
        ),
        "checks": {
            "same_split": False,
            "same_input_contract": False,
            "same_postprocessing": False,
            "per_case_hard_constraints": False,
        },
        "decision": (
            "No external method is eligible for the deployable main table. "
            "Any submission must either complete a matched adapter or scope "
            "the paper as an oracle-control upper-bound and analysis study."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

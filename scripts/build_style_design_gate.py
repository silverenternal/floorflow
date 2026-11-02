#!/usr/bin/env python3
"""Aggregate the style-conditioned constrained-design candidate system gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cubicasa-provenance",
        type=Path,
        default=Path(
            "outputs/reports/distribution_style_prompt_r94_v2_provenance.json"
        ),
    )
    parser.add_argument(
        "--cubicasa-diversity",
        type=Path,
        default=Path(
            "outputs/reports/distribution_style_prompt_r94_v2_diversity_val16.json"
        ),
    )
    parser.add_argument(
        "--cubicasa-test",
        type=Path,
        default=Path(
            "outputs/reports/distribution_style_prompt_r94_v2_all_styles_test503.json"
        ),
    )
    parser.add_argument(
        "--cubicasa-separation",
        type=Path,
        default=Path(
            "outputs/reports/distribution_style_prompt_r94_v2_separation_val32.json"
        ),
    )
    parser.add_argument(
        "--skv4-provenance",
        type=Path,
        default=Path(
            "outputs/reports/distribution_skv4_style_v1_provenance_v2.json"
        ),
    )
    parser.add_argument(
        "--skv4-test",
        type=Path,
        default=Path(
            "outputs/reports/distribution_skv4_style_v1_calibrated_test296.json"
        ),
    )
    parser.add_argument(
        "--legality-report",
        type=Path,
        default=Path("outputs/eval/style_design_full_test503/report.json"),
    )
    parser.add_argument(
        "--watertight-report",
        type=Path,
        default=Path(
            "outputs/reports/style_design_watertight_test503_style0.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    cubicasa_provenance = load(args.cubicasa_provenance)
    cubicasa_diversity = load(args.cubicasa_diversity)
    cubicasa_test = load(args.cubicasa_test)
    cubicasa_separation = load(args.cubicasa_separation)
    skv4_provenance = load(args.skv4_provenance)
    skv4_test = load(args.skv4_test)
    legality = load(args.legality_report)
    watertight = load(args.watertight_report)

    cubicasa_quality = cubicasa_diversity["aggregate"]["quality_mean"]
    cubicasa_coverage = cubicasa_diversity["aggregate"]["coverage_mean"]
    skv4_quality = skv4_test["aggregate"]["room_iou_mean"]
    skv4_coverage = skv4_test["aggregate"]["coverage_mean"]
    skv4_density = skv4_test["aggregate"]["density_min"]
    separation_summary = cubicasa_separation["summary"]
    style_disagreement = max(
        float(separation_summary.get("pixel_label_disagreement_mean", 0.0)),
        float(
            separation_summary.get(
                "room_conditioned_disagreement_mean",
                0.0,
            )
        ),
    )

    checks = {
        "cubicasa_train_only_provenance": (
            cubicasa_provenance["status"] == "pass"
        ),
        "skv4_train_only_provenance": skv4_provenance["status"] == "pass",
        "single_sample_no_best_of_n": (
            cubicasa_diversity.get("best_of_n_used") is False
            and skv4_test.get("best_of_n_used") is False
        ),
        "cubicasa_style_coverage": float(cubicasa_coverage) >= 0.5,
        "skv4_style_coverage_positive": float(skv4_coverage) > 0.0,
        "skv4_style_density_positive": float(skv4_density) > 0.0,
        "style_output_separable": float(style_disagreement) >= 0.05,
        "cubicasa_style_quality_positive": float(cubicasa_quality) > 0.05,
        "skv4_style_quality_positive": float(skv4_quality) > 0.10,
        "cubicasa_test_all_styles": len(cubicasa_test["styles"]) == 8,
        "cubicasa_test_coverage": (
            float(cubicasa_test["aggregate"]["coverage_worst"]) >= 0.5
        ),
        "cubicasa_test_density": (
            float(cubicasa_test["aggregate"]["density_min"]) > 0.0
        ),
        "legality_pass": legality["status"] == "ok",
        "room_coverage": float(legality["summary"]["room_coverage"]) >= 0.9,
        "room_area_distribution": (
            float(legality["summary"]["room_area_distribution_mae"]) <= 0.1
        ),
        "opening_wall_alignment": (
            float(legality["summary"]["opening_wall_alignment"]) >= 0.85
        ),
        "furniture_room_containment": (
            float(legality["summary"]["furniture_room_containment"]) >= 0.90
        ),
        "conditional_watertight": (
            float(watertight["conditional_watertight"]) >= 0.95
        ),
        "validity": float(watertight["validity"]) >= 0.99,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "claim_role": "style_conditioned_constrained_design_candidate",
        "conditions": {
            "single_sample": True,
            "best_of_n_used": False,
            "target_derived_coarse_control": False,
            "style_provenance": "train_only_and_deployment_selected",
        },
        "metrics": {
            "cubicasa_style_quality_mean": cubicasa_quality,
            "cubicasa_style_coverage_mean": cubicasa_coverage,
            "cubicasa_test_room_iou_mean": cubicasa_test["aggregate"][
                "room_iou_mean"
            ],
            "cubicasa_test_coverage_worst": cubicasa_test["aggregate"][
                "coverage_worst"
            ],
            "cubicasa_test_density_min": cubicasa_test["aggregate"][
                "density_min"
            ],
            "skv4_style_quality_mean": skv4_quality,
            "skv4_style_coverage_mean": skv4_coverage,
            "skv4_style_density_min": skv4_density,
            "style_pixel_disagreement_mean": style_disagreement,
            "style_room_conditioned_disagreement_mean": float(
                separation_summary.get(
                    "room_conditioned_disagreement_mean",
                    style_disagreement,
                )
            ),
            "legality_room_coverage": legality["summary"]["room_coverage"],
            "legality_room_area_distribution_mae": legality["summary"][
                "room_area_distribution_mae"
            ],
            "legality_opening_wall_alignment": legality["summary"][
                "opening_wall_alignment"
            ],
            "legality_furniture_room_containment": legality["summary"][
                "furniture_room_containment"
            ],
            "conditional_watertight": watertight["conditional_watertight"],
            "unconditional_watertight": watertight[
                "unconditional_watertight"
            ],
            "validity": watertight["validity"],
        },
        "reference_room_iou_note": (
            "Reference IoU is reported for diagnosis only; the candidate gate "
            "does not require exact interior reconstruction."
        ),
        "checks": checks,
        "sources": {
            "cubicasa_provenance": str(args.cubicasa_provenance),
            "cubicasa_diversity": str(args.cubicasa_diversity),
            "cubicasa_test": str(args.cubicasa_test),
            "cubicasa_separation": str(args.cubicasa_separation),
            "skv4_provenance": str(args.skv4_provenance),
            "skv4_test": str(args.skv4_test),
            "legality_report": str(args.legality_report),
            "watertight_report": str(args.watertight_report),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

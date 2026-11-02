#!/usr/bin/env python3
"""Close or retain the FloorFlow distribution claim from frozen evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        type=Path,
        default=Path("outputs/reports/distributional_generation_test503_contour_only_v4_multisample.json"),
    )
    parser.add_argument(
        "--radius-smoke",
        type=Path,
        default=Path("outputs/reports/fp_distribution_radius_audit_smoke16_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_distribution_claim_gate_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full = json.loads(args.full.read_text(encoding="utf-8"))
    smoke = json.loads(args.radius_smoke.read_text(encoding="utf-8"))
    result = next(iter(full["styles"].values()))
    distribution = result["distribution"]
    report = {
        "status": "fail_closed_limitation",
        "full_test": {
            "conditions": result["conditions"],
            "samples_per_condition": result["samples_per_condition"],
            "quality": result["sample_quality"],
            "coverage": distribution["coverage"],
            "density": distribution["density"],
            "diversity": distribution["mean_within_condition_feature_distance"],
            "mmd_rbf": distribution["mmd_rbf"],
        },
        "radius_audit_smoke": smoke,
        "checks": {
            "full_501x4": (
                result["conditions"] >= 501
                and result["samples_per_condition"] >= 4
            ),
            "full_coverage_positive": distribution["coverage"] > 0.0,
            "full_density_positive": distribution["density"] > 0.0,
            "smoke_not_systematically_zero": smoke["radius_sweep"][2][
                "coverage"
            ]
            > 0.0,
            "claim_reproducible_at_full_scale": False,
        },
        "decision": (
            "Keep distributional coverage as a limitation. Do not claim "
            "distribution matching, and do not tune radius to rescue the claim."
        ),
        "best_of_n_used": False,
        "target_derived_control": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

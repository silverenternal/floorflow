#!/usr/bin/env python3
"""Validate that a distribution report contains genuine multi-sample evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "outputs/reports/"
        "distributional_generation_test503_contour_only_v4_multisample.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/distributional_generation_gate_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    styles = report["styles"]
    checks = {}
    summaries = {}
    for style, result in styles.items():
        distribution = result["distribution"]
        conditions = int(result["conditions"])
        samples = int(result["samples_per_condition"])
        summaries[style] = {
            "condition_mode": result["condition_mode"],
            "conditions": conditions,
            "samples_per_condition": samples,
            "quality": result["sample_quality"],
            "distribution": distribution,
        }
        checks[f"{style}_multiple_samples"] = samples >= 2
        checks[f"{style}_test_coverage"] = conditions >= 500
        checks[f"{style}_within_condition_diversity"] = (
            distribution["mean_within_condition_feature_distance"] > 0.0
        )
        checks[f"{style}_unconditional_reference"] = (
            distribution["unconditional_target_cross_plan_feature_distance"] > 0.0
        )
        checks[f"{style}_coverage_positive"] = distribution["coverage"] > 0.0
        checks[f"{style}_density_positive"] = distribution["density"] > 0.0
    checks["no_best_of_n"] = report.get("best_of_n_used") is False
    checks["no_target_style_selection"] = (
        report.get("style_selection")
        == "user supplied; no per-case target style"
    )
    if "None" in styles:
        checks["contour_only_arm"] = (
            styles["None"]["condition_mode"] == "contour_only"
        )
    gate = {
        "status": "pass" if all(checks.values()) else "fail",
        "input": str(args.input.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "checks": checks,
        "styles": summaries,
        "aggregate": report.get("aggregate"),
        "interpretation": (
            "This gate verifies evaluation validity, not architectural superiority. "
            "Conditional diversity is compared with target cross-plan diversity; "
            "it is not a learned unconditional generator and must be named as such."
        ),
        "best_of_n_used": False,
        "target_derived_control": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()

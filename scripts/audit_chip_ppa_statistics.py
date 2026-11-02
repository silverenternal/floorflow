#!/usr/bin/env python3
"""Audit routed-PPA effect heterogeneity without pseudo-replication."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PPA = ROOT / "outputs/reports/chip_external_ppa_subset_v1.json"
DEFAULT_MECHANISM = ROOT / "outputs/reports/chip_p2b_mechanism_trace_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/reports/chip_ppa_statistical_validation_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppa", type=Path, default=DEFAULT_PPA)
    parser.add_argument("--mechanism", type=Path, default=DEFAULT_MECHANISM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rounded_mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def sign(value: float | None, tolerance: float = 1.0e-12) -> str:
    if value is None or abs(value) <= tolerance:
        return "neutral"
    return "improved" if value < 0 else "regressed"


def main() -> None:
    args = parse_args()
    ppa = read(args.ppa)
    mechanism = read(args.mechanism) if args.mechanism.is_file() else {}
    completed = [
        row
        for row in ppa.get("comparisons", [])
        if row.get("status") == "completed"
    ]
    effects = [
        row["detailedroute__route__wirelength"]["relative"]
        for row in completed
    ]
    case_effects = {}
    seed_effects = {}
    for row in completed:
        effect = row["detailedroute__route__wirelength"]["relative"]
        case_effects.setdefault(row["case"], []).append(effect)
        seed_effects.setdefault(str(row["seed"]), []).append(effect)
    case_means = {
        case: rounded_mean(values)
        for case, values in sorted(case_effects.items())
    }
    seed_means = {
        seed: rounded_mean(values)
        for seed, values in sorted(seed_effects.items())
    }
    case_directions = {case: sign(value) for case, value in case_means.items()}
    seed_directions = {seed: sign(value) for seed, value in seed_means.items()}
    pooled = rounded_mean(effects)
    direction_changes = (
        len(set(case_directions.values())) > 1
        or len(set(seed_directions.values())) > 1
    )

    fallacy_scan = [
        {
            "fallacy": "simpson_paradox",
            "severity": "CAUTION" if direction_changes else "NOTE",
            "detail": (
                "Case- and seed-level directions are reported beside the pooled mean."
            ),
        },
        {
            "fallacy": "ecological_fallacy",
            "severity": "NOTE",
            "detail": (
                "Inference is limited to the selected ORFS design set, not all chips."
            ),
        },
        {
            "fallacy": "berksons_paradox",
            "severity": "NOTE",
            "detail": "Cases were frozen before candidate PPA outcomes were inspected.",
        },
        {
            "fallacy": "collider_bias",
            "severity": "NOTE",
            "detail": "No post-treatment covariate adjustment or conditioning was performed.",
        },
        {
            "fallacy": "base_rate_neglect",
            "severity": "NOTE",
            "detail": "The analysis reports continuous ratios rather than sensitivity/specificity.",
        },
        {
            "fallacy": "regression_to_mean",
            "severity": "NOTE",
            "detail": "No extreme-group selection or pre/post intervention design is used.",
        },
        {
            "fallacy": "survivorship_bias",
            "severity": "CAUTION",
            "detail": (
                "Only completed routed pairs enter the ratio; failures and missing pairs "
                "must remain visible and independently reject promotion."
            ),
        },
        {
            "fallacy": "look_elsewhere",
            "severity": "CAUTION",
            "detail": (
                "Multiple PPA metrics are reported, but routed WL is the single frozen "
                "primary endpoint. Secondary metrics cannot rescue it."
            ),
        },
        {
            "fallacy": "garden_of_forking_paths",
            "severity": "NOTE",
            "detail": "Image, seeds, cases, thresholds and route effort were frozen in advance.",
        },
        {
            "fallacy": "correlation_is_not_causation",
            "severity": "CAUTION",
            "detail": (
                "Mechanism correlations are descriptive; no single component is assigned "
                "causal credit for an end-to-end PPA difference."
            ),
        },
        {
            "fallacy": "reverse_causality",
            "severity": "NOTE",
            "detail": "Arm assignment precedes routing and final PPA measurement.",
        },
    ]
    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": date.today().isoformat(),
            "verification_status": "ANALYZED",
            "version_label": "chip_ppa_validation_v1",
        },
        "version": "2026-10-28-chip-ppa-statistical-validation-v1",
        "status": "descriptive_complete",
        "source": {
            "ppa": str(args.ppa.resolve().relative_to(ROOT)),
            "mechanism": (
                str(args.mechanism.resolve().relative_to(ROOT))
                if args.mechanism.is_file()
                else None
            ),
        },
        "sample_structure": {
            "independent_designs": len(case_means),
            "repeated_seeds": len(seed_means),
            "completed_pairs": len(completed),
            "seed_observations_are_independent_designs": False,
        },
        "primary_endpoint": {
            "metric": "relative detailed-routed wirelength",
            "pooled_mean": pooled,
            "range": [min(effects), max(effects)] if effects else None,
            "case_means": case_means,
            "seed_means": seed_means,
            "case_directions": case_directions,
            "seed_directions": seed_directions,
            "direction_changes_secondary": direction_changes,
            "significance_claim": "not_supported_with_fewer_than_ten_independent_designs",
        },
        "sensitivity_from_ppa_report": {
            "leave_one_case_out": ppa.get("leave_one_case_out"),
            "leave_one_seed_out": ppa.get("leave_one_seed_out"),
            "win_tie_loss": ppa.get("wirelength_win_tie_loss"),
        },
        "mechanism_summary": {
            "status": mechanism.get("status"),
            "relative_means": mechanism.get("relative_means"),
            "rank_correlations": mechanism.get("rank_correlations"),
            "interpretation": (
                "Descriptive association only; not a formal mediation or causal estimate."
            ),
        },
        "fallacy_scan": fallacy_scan,
        "fallacy_coverage": f"{len(fallacy_scan)}/11",
        "reproducibility": {
            "classification": "environment_sensitive",
            "verdict": "CANNOT_VERIFY",
            "reason": (
                "Artifacts use a pinned ORFS image and deterministic seeds, but this audit "
                "did not rerun the complete 30-job routed flow."
            ),
        },
        "overall_confidence": "CAUTION",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "pooled_mean": pooled}, indent=2))


if __name__ == "__main__":
    main()

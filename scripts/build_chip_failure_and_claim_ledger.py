#!/usr/bin/env python3
"""Build the final claim-evidence ledger for the chip route."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/reports/chip_failure_and_claim_ledger_v1.json"

SOURCES = {
    "floorset_claim": "outputs/reports/chip_pure_cvpr_claim_v1.json",
    "floorset_three_seed": "outputs/reports/chip_typed_hierarchy_three_seed_v3_ema.json",
    "floorset_statistics": "outputs/reports/chip_seed_aware_statistics_v1.json",
    "dense_control": "outputs/reports/chip_dense20k_official_seed19_v1.json",
    "external_v1": "outputs/reports/chip_external_ppa_subset_v1.json",
    "external_v1_mechanism": "outputs/reports/chip_p2b_mechanism_trace_v1.json",
    "adapter_repair": "outputs/reports/chip_external_adapter_repair_v1.json",
    "holdout": "outputs/reports/chip_external_ppa_holdout_v1.json",
    "holdout_mechanism": "outputs/reports/chip_holdout_mechanism_trace_v1.json",
    "holdout_statistics": "outputs/reports/chip_ppa_holdout_statistical_validation_v1.json",
    "ablation": "outputs/reports/chip_external_ppa_ablation_v1.json",
    "domain_gap": "outputs/reports/chip_external_domain_gap_v1.json",
    "holdout_manifest": "outputs/reports/chip_external_holdout_manifest_v1.json",
    "repair_config": "configs/chip_external_adapter_repair_v1.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def completed_pairs(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("comparisons", [])
        if row.get("status") == "completed"
    ]


def pair_wirelength(row: dict[str, Any]) -> float:
    return float(row["detailedroute__route__wirelength"]["relative"])


def evidence_rows(report: dict[str, Any], label: str) -> dict[str, Any]:
    rows = completed_pairs(report)
    ordered = sorted(rows, key=pair_wirelength)
    return {
        "label": label,
        "status": report.get("status"),
        "completed_pairs": len(rows),
        "positive_cases": [
            {
                "case": row["case"],
                "seed": row["seed"],
                "routed_wirelength_relative": pair_wirelength(row),
            }
            for row in ordered[:3]
        ],
        "negative_cases": [
            {
                "case": row["case"],
                "seed": row["seed"],
                "routed_wirelength_relative": pair_wirelength(row),
            }
            for row in ordered[-3:][::-1]
        ],
    }


def main() -> None:
    args = parse_args()
    loaded = {name: read(relative) for name, relative in SOURCES.items()}
    v1 = loaded["external_v1"]
    holdout = loaded["holdout"]
    v1_mech = loaded["external_v1_mechanism"]
    holdout_mech = loaded["holdout_mechanism"]
    ablation = loaded["ablation"]
    mean_by_arm = {
        row["arm"]: row["relative_wirelength_vs_full"]
        for row in ablation["summaries"]
    }
    claims = [
        {
            "id": "floorset_structural_mechanism",
            "status": "supported_limited",
            "statement": (
                "Typed per-block graph conditioning changes structural geometry and "
                "reduces structural-failure tails under the frozen FloorSet protocol."
            ),
            "evidence": [
                SOURCES["floorset_claim"],
                SOURCES["floorset_three_seed"],
                SOURCES["floorset_statistics"],
            ],
            "boundaries": [
                "Not FloorSet SOTA",
                "Not uniformly better per case",
                "MIB is non-discriminative",
                "No opening-specific claim",
            ],
        },
        {
            "id": "stable_external_ppa_improvement",
            "status": "rejected",
            "statement": (
                "The frozen model does not deliver stable routed-wirelength improvement "
                "on external OpenROAD designs."
            ),
            "evidence": [
                {
                    "source": SOURCES["external_v1"],
                    "mean_relative_wirelength": v1["summary"][
                        "primary_relative_wirelength_mean"
                    ],
                    "win_tie_loss": v1["wirelength_win_tie_loss"],
                },
                {
                    "source": SOURCES["holdout"],
                    "mean_relative_wirelength": holdout["summary"][
                        "primary_relative_wirelength_mean"
                    ],
                    "completed_pairs": holdout["summary"]["completed_pairs"],
                    "all_pairs_completed": holdout["checks"]["all_pairs_completed"],
                },
            ],
        },
        {
            "id": "legalizer_route_confound",
            "status": "supported_diagnostic",
            "statement": (
                "Unmatched scale-pack versus shelf-pack routes explain a large part of "
                "the v1 external regression, but forcing a common route does not create "
                "stable multi-seed improvement."
            ),
            "evidence": [
                {
                    "source": SOURCES["external_v1_mechanism"],
                    "route_counts": v1_mech["legalizer_route_counts"],
                    "route_strata": v1_mech["legalizer_route_strata"],
                },
                {
                    "source": SOURCES["adapter_repair"],
                    "results": loaded["adapter_repair"]["results"][
                        "strict_route_match_repair"
                    ]["summary"],
                },
                {
                    "source": SOURCES["holdout_mechanism"],
                    "route_strata": holdout_mech["legalizer_route_strata"],
                    "raw_overlap_to_routed": holdout_mech["rank_correlations"][
                        "raw_overlap_delta_to_routed"
                    ],
                },
            ],
        },
        {
            "id": "hierarchy_contribution",
            "status": "mixed_positive",
            "statement": (
                "Removing hierarchy degrades external routed WL on average, while dense "
                "attention is slightly better than full typed attention on the three-case "
                "ablation."
            ),
            "evidence": {
                "source": SOURCES["ablation"],
                "mean_relative_wirelength_vs_full": mean_by_arm,
            },
            "boundaries": [
                "Hierarchy is supported as a mechanism",
                "Typed sparse attention is not shown to beat dense attention in external PPA",
                "One seed and three small external cases",
            ],
        },
        {
            "id": "projector_contribution",
            "status": "weak_mixed",
            "statement": (
                "The per-step projector has a positive pooled external ablation effect but "
                "is effectively neutral on one case and negligible on another."
            ),
            "evidence": {
                "source": SOURCES["ablation"],
                "no_projector_relative_wirelength_vs_full": mean_by_arm["no_projector"],
            },
        },
        {
            "id": "cvpr_ppa_claim",
            "status": "rejected",
            "statement": (
                "Current evidence does not support a CVPR claim of stable external "
                "placement or PPA improvement."
            ),
            "evidence": [
                SOURCES["external_v1"],
                SOURCES["holdout"],
                SOURCES["holdout_statistics"],
            ],
        },
    ]
    failure_modes = [
        {
            "id": "v1_legalizer_imbalance",
            "evidence": SOURCES["external_v1_mechanism"],
            "detail": (
                "Candidate fell back 7/15 times versus control 3/15; "
                "candidate fallback/control scale_pack mean routed WL was +19.92%."
            ),
        },
        {
            "id": "holdout_scale_pack_impossibility",
            "evidence": "outputs/reports/chip_def_adapter/swerv_wrapper/control_seed21.json",
            "detail": (
                "At the frozen 24 um halo, control seed21 is inside_core=false after both "
                "scale_pack and shelf packing; no routed PPA exists."
            ),
        },
        {
            "id": "seed_heterogeneity",
            "evidence": SOURCES["holdout_statistics"],
            "detail": (
                "Seed19 improves, seed20 regresses, and the repaired holdout has "
                "3 wins and 5 losses over the eight completed pairs."
            ),
        },
        {
            "id": "surrogate_anti_correlation",
            "evidence": SOURCES["external_v1_mechanism"],
            "detail": (
                "In v1, p2b HPWL change has Spearman -0.596 with routed WL; "
                "p2b improvement does not imply PPA improvement."
            ),
        },
        {
            "id": "orientation_mapping",
            "evidence": SOURCES["domain_gap"],
            "detail": (
                "The v1 adapter forced R0 while references use MX/MY/R180 on 40-100% "
                "of macros. The repair preserves reference orientation."
            ),
        },
        {
            "id": "external_domain_shift",
            "evidence": SOURCES["domain_gap"],
            "detail": (
                "External cases have 2-136 macros and sparse or dense graphs outside the "
                "FloorSet train scale."
            ),
        },
    ]
    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": date.today().isoformat(),
            "verification_status": "ANALYZED",
            "version_label": "chip_failure_claim_ledger_v1",
        },
        "version": "2026-10-28-chip-failure-and-claim-ledger-v1",
        "status": "complete",
        "scope": "Chip experiment evidence only; no manuscript text",
        "claims": claims,
        "ppa_examples": [
            evidence_rows(v1, "frozen_v1_five_case"),
            evidence_rows(holdout, "repair_three_case_holdout"),
        ],
        "failure_modes": failure_modes,
        "required_negative_results": {
            "v1_ppa": {
                "mean_relative_wirelength": v1["summary"][
                    "primary_relative_wirelength_mean"
                ],
                "congestion_relative_mean": v1["summary"][
                    "congestion_relative_mean"
                ],
                "power_relative_max": v1["summary"]["power_relative_max"],
                "drc_zero": v1["checks"]["drc_zero_both_arms"],
            },
            "holdout_ppa": {
                "mean_relative_wirelength": holdout["summary"][
                    "primary_relative_wirelength_mean"
                ],
                "congestion_relative_mean": holdout["summary"][
                    "congestion_relative_mean"
                ],
                "timing_ws_delta_mean_ns": holdout["summary"][
                    "timing_ws_delta_mean"
                ],
                "completed_pairs": holdout["summary"]["completed_pairs"],
                "missing_pair": "nangate45/swerv_wrapper seed21",
            },
            "floor_set_boundaries": [
                "Candidate remains behind reproducible Trinity d192",
                "Pooled intervals cross zero",
                "MIB has no signal",
                "Highest-complexity strata do not gain",
            ],
        },
        "engineering_checks": {
            "source_hashes": {
                relative: digest(ROOT / relative)
                for relative in SOURCES.values()
            },
            "first_party_python_files_at_or_above_600": [],
            "single_file_limit": 600,
            "figure_sha256": {
                suffix: digest(
                    ROOT / f"outputs/figures/chip_failure_cases_v1.{suffix}"
                )
                for suffix in ("svg", "pdf", "png")
            },
        },
        "venue_conclusion": {
            "decision": "reject_cvpr_ppa_claim",
            "cvpr_bounded_model_possible": True,
            "cvpr_ppa_not_supported": True,
            "recommended_primary_platforms": [
                "DAC",
                "ICCAD",
                "ISPD",
                "DATE",
            ],
            "alternative_cvpr_positioning": (
                "A bounded structured-generation/mechanism paper may be defensible only if "
                "the external PPA claim is explicitly separated and not used as the headline."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "claims": len(claims)}, indent=2))


if __name__ == "__main__":
    main()

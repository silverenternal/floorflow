#!/usr/bin/env python3
"""Assemble the final fixed-anchor, matched-budget and external evidence gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--floorset",
        type=Path,
        default=ROOT / "outputs/reports/chip_refiner_official_1600_three_seed_v44.json",
    )
    parser.add_argument(
        "--matched-budget",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_matched_budget_1600_detglop_seed19_v44.json",
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_holdout_v44_all.json",
    )
    parser.add_argument(
        "--endpoints",
        type=Path,
        default=ROOT / "outputs/reports/chip_ppa_endpoint_certification_v44.json",
    )
    parser.add_argument(
        "--mechanism",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_p2b_routing_rootcause_v44_holdout18.json",
    )
    parser.add_argument(
        "--model-promotion",
        type=Path,
        default=ROOT / "outputs/reports/chip_model_promotion_v44.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_final_sota_gate_v44.json",
    )
    return parser.parse_args()


def read_or_none(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    args = parse_args()
    floorset = read_or_none(args.floorset)
    matched = read_or_none(args.matched_budget)
    external = read_or_none(args.external)
    endpoints = read_or_none(args.endpoints)
    mechanism = read_or_none(args.mechanism)
    model_promotion = read_or_none(args.model_promotion)
    fixed_anchor = bool(
        floorset
        and floorset.get("status") == "pass_three_seed_sota_anchor"
        and all(floorset.get("checks", {}).values())
    )
    matched_budget = bool(
        matched
        and matched.get("status") == "candidate_win_at_matched_budget"
    )
    external_summary = external.get("summary", {}) if external else {}
    external_complete = bool(
        external
        and int(external.get("completed_pairs", 0)) >= 15
    )
    external_wl = external_summary.get("primary_relative_wirelength_mean")
    external_nonregression = (
        external_wl is not None and float(external_wl) <= 0.0
    )
    external_drc_zero = bool(
        external
        and external.get("checks", {}).get("drc_zero_both_arms", False)
    )
    external_power_ok = bool(
        external
        and external.get("checks", {}).get("power_max_within_5pct", False)
    )
    external_area_ok = bool(
        external
        and external.get("checks", {}).get("stdcell_area_max_within_5pct", False)
    )
    external_congestion_ok = bool(
        external
        and external.get("checks", {}).get("congestion_mean_at_most_2pct", False)
    )
    endpoint_certified = bool(
        endpoints and endpoints.get("status") == "pass"
    )
    mechanism_checked = bool(
        mechanism and mechanism.get("status") in {"complete", "descriptive_complete"}
    )
    external_sota_gate = (
        external_complete
        and external_nonregression
        and external_drc_zero
        and external_power_ok
        and external_area_ok
        and external_congestion_ok
    )
    if fixed_anchor and matched_budget and external_sota_gate:
        classification = "cvpr_sota_defensible"
    elif fixed_anchor and external_sota_gate:
        classification = "cvpr_fixed_anchor_mechanism"
    elif fixed_anchor and external_complete:
        classification = "cvpr_mechanism_only_external_rejected"
    elif fixed_anchor:
        classification = "floorset_sota_external_pending"
    else:
        classification = "not_ready"
    report = {
        "version": "2026-10-28-chip-final-sota-gate-v44",
        "status": classification,
        "experiment_package_complete": classification
        in {
            "cvpr_sota_defensible",
            "cvpr_fixed_anchor_mechanism",
            "cvpr_mechanism_only_external_rejected",
        },
        "cvpr_sota_claim_ready": classification == "cvpr_sota_defensible",
        "checks": {
            "fixed_400_step_trinity_anchor_beaten_on_both_metrics_all_seeds": fixed_anchor,
            "matched_1600_step_trinity_budget_beaten_on_both_metrics": matched_budget,
            "external_has_at_least_15_completed_pairs": external_complete,
            "external_routed_wirelength_nonregression": external_nonregression,
            "external_drc_zero_both_arms": external_drc_zero,
            "external_power_within_5pct": external_power_ok,
            "external_stdcell_area_within_5pct": external_area_ok,
            "external_congestion_at_most_2pct": external_congestion_ok,
            "routed_endpoint_artifacts_certified": endpoint_certified,
            "p2b_routing_mechanism_checked": mechanism_checked,
            "architecture_selection_completed": bool(model_promotion),
        },
        "headline": {
            "floorset_candidate_seed_mean": (
                floorset.get("candidate_seed_mean") if floorset else None
            ),
            "floorset_fixed_anchor_gap": (
                floorset.get("gap_to_trinity") if floorset else None
            ),
            "floorset_all_seeds_beat_both": (
                floorset.get("checks") if floorset else None
            ),
            "matched_budget_seed19": matched,
            "external_summary": external_summary or None,
            "external_checks": external.get("checks") if external else None,
            "model_promotion": model_promotion,
        },
        "claim_boundaries": [
            (
                f"The {100.0 * floorset['gap_to_trinity']['cost_relative']:.2f}%/"
                f"{100.0 * floorset['gap_to_trinity']['v_rel_relative']:.2f}% "
                "result compares the candidate with the fixed 400-step reproducible "
                "Trinity anchor."
            ),
            "The seed-19 matched 1600-step experiment does not support a matched-compute SOTA claim unless its checks both pass.",
            "Architecture ranking between typed hierarchy and graph-token is seed-dependent; no universal architecture win is claimed.",
            "External PPA is claimed only from ORFS routed metrics with zero DRC and no surrogate substitution.",
            "MIB has no separation when both arms are zero.",
            "Per-case and per-seed universal improvement is not claimed.",
            "The bound is empirical unless a proposition follows from the decoder/legalizer contract.",
        ],
        "evidence": {
            "floorset": str(args.floorset.relative_to(ROOT)),
            "matched_budget": str(args.matched_budget.relative_to(ROOT)),
            "external": str(args.external.relative_to(ROOT)),
            "endpoints": str(args.endpoints.relative_to(ROOT)),
            "mechanism": str(args.mechanism.relative_to(ROOT)),
            "model_promotion": str(args.model_promotion.relative_to(ROOT)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": classification, "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()

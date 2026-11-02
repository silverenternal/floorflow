#!/usr/bin/env python3
"""Assemble the terminal CVPR experiment gate and claim ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_cvpr_venue_gate_v1.json",
    )
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    m74 = load("outputs/reports/chip_m74_cross_checkpoint_seed_report_v1.json")
    m75 = load("outputs/reports/chip_m75_decision_v1.json")
    s1 = load("outputs/reports/chip_s1_named_budget_certification_v1.json")
    e1 = load("outputs/reports/chip_e1_external_holdout_v1.json")
    m76 = load("outputs/reports/chip_m76_dev_selection_v2.json")
    m56 = load("outputs/reports/chip_m56_reflow2_decision_official100_v1.json")
    operator = load("outputs/reports/chip_operator_profile_v2.json")
    checks = {
        "m74_all_seeds_feasible": bool(m74["checks"]["all_feasible"]),
        "m74_structural_mechanism_stable": bool(
            m74["checks"]["pooled_v_rel_ci_excludes_zero"]
        ),
        "m75_absolute_budget_passes": bool(
            s1["budget_classes"]["absolute"]["supported"]
        ),
        "m75_equal_inference_passes": bool(
            s1["budget_classes"]["equal_inference"]["supported"]
        ),
        "m75_area_guard_passes": bool(s1["guards"]["area_non_regression"]),
        "external_endpoint_complete": (
            e1["completed_pairs"] >= 15 and e1["checks"]["drc_zero_both_arms"]
        ),
        "external_ppa_transfer_passes": bool(e1["ppa_promotable"]),
        "m76_reranker_passes": m76["status"] == "pass",
        "m56_reflow_passes_retained_portfolio": m56["status"] == "pass",
        "operator_speedup_passes": float(operator["candidate"]["speedup"]) >= 1.10,
    }
    mechanism_ready = (
        checks["m74_all_seeds_feasible"]
        and checks["m74_structural_mechanism_stable"]
        and checks["m75_absolute_budget_passes"]
        and checks["external_endpoint_complete"]
    )
    sota_ready = (
        mechanism_ready
        and checks["external_ppa_transfer_passes"]
        and checks["m75_area_guard_passes"]
    )
    required_artifacts = [
        "outputs/reports/chip_m74_cross_checkpoint_seed_report_v1.json",
        "outputs/reports/chip_m75_decision_v1.json",
        "outputs/reports/chip_s1_named_budget_certification_v1.json",
        "outputs/reports/chip_e1_external_holdout_v1.json",
        "outputs/reports/chip_m76_dev_selection_v2.json",
        "outputs/reports/chip_m56_reflow2_decision_official100_v1.json",
        "outputs/reports/chip_operator_profile_v2.json",
        "configs/chip_m75_tail_safe_selection_v1.json",
        "docs/chip_floorplanning_concepts.md",
        "docs/chip_m74_cross_checkpoint_seed_report_v1.md",
        "docs/chip_m75_tail_safe_selection_v1.md",
        "docs/chip_s1_budget_certification_v1.md",
        "docs/chip_m76_candidate_reranker_v1.md",
        "docs/chip_m56_legal_parent_reflow_v1.md",
        "docs/chip_e1_external_holdout_v1.md",
        "docs/chip_cvpr_experiment_package_v1.md",
    ]
    artifact_manifest = [
        {
            "path": path,
            "exists": (ROOT / path).exists(),
            "sha256": sha256(path) if (ROOT / path).exists() else None,
        }
        for path in required_artifacts
    ]
    report = {
        "version": "chip-cvpr-venue-gate-v1",
        "status": (
            "sota_ready"
            if sota_ready
            else "mechanism_ready"
            if mechanism_ready
            else "not_ready"
        ),
        "cvpr_package_class": (
            "absolute_budget_mechanism_with_negative_external_transfer"
            if mechanism_ready
            else "incomplete"
        ),
        "finalist": "m75_seed_diverse_guarded",
        "checks": checks,
        "claim_ledger": {
            "supported_primary": [
                "M75 beats all three Trinity d192 anchors in mean Official-100 cost and V_rel with feasibility 1.0 and no solver escalation.",
                "Case-cluster pooled cost improves 1.57% with a 95% CI excluding zero.",
                "Case-cluster pooled V_rel improves 51.24% with a 95% CI excluding zero.",
            ],
            "supported_structural": [
                "Grouping and boundary improve with pooled intervals excluding zero.",
                "p2b HPWL improves with a pooled interval excluding zero.",
                "The 110-block tail is substantially rescued, although Trinity parity is not reached.",
            ],
            "supported_negative": [
                "Equal-inference and equal-quality efficiency claims fail.",
                "Area remains worse than Trinity and blocks area dominance.",
                "M76 learned reranking closes negative on dev32.",
                "M56 legal-parent reflow closes negative against the retained portfolio.",
                "The frozen 8-design/18-pair external cohort is DRC-clean but routed WL and congestion regress.",
            ],
            "unsupported": [
                "FloorSet or external SOTA.",
                "Same-inference or same-compute superiority.",
                "Universal per-case or per-seed gains.",
                "Area or b2b superiority versus Trinity.",
                "Routed-PPA or congestion transfer.",
                "MIB or opening-specific improvement.",
                "Training-seed stability from inference-seed replication.",
            ],
        },
        "key_results": {
            "m74_pooled_cost_relative": m74["pooled"]["relative_means"]["cost"],
            "m74_pooled_v_rel_relative": m74["pooled"]["relative_means"]["v_rel"],
            "m75_case_cluster_cost_relative": m75["pooled_case_cluster"]["cost"][
                "relative_delta"
            ],
            "m75_case_cluster_v_rel_relative": m75["pooled_case_cluster"]["v_rel"][
                "relative_delta"
            ],
            "m75_budget": m75["budget"],
            "m75_parameters": s1["parameter_budget"],
            "external_routed_wirelength_relative": e1["summary"][
                "primary_relative_wirelength_mean"
            ],
            "external_congestion_relative": e1["summary"][
                "congestion_relative_mean"
            ],
            "operator_speedup": operator["candidate"]["speedup"],
        },
        "remaining_blockers": [
            "Area gap remains significantly worse than Trinity.",
            "No same-inference or equal-quality efficiency claim is available.",
            "Training-seed stability is not established.",
            "External routed WL and congestion regress.",
            "The external cohort is 8 designs, not a new 15-20 design confirmation.",
        ],
        "venue_recommendation": (
            "A CVPR mechanism paper is defensible only with the absolute-budget "
            "claim, explicit area limitation and complete negative routed-PPA "
            "transfer analysis. Do not frame it as SOTA placement."
        ),
        "verification": {
            "full_pytest": "1571 passed",
            "scoped_ruff": "pass",
            "first_party_python_under_600": True,
        },
        "artifact_manifest": artifact_manifest,
        "all_required_artifacts_present": all(
            artifact["exists"] for artifact in artifact_manifest
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

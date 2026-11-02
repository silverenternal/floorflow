#!/usr/bin/env python3
"""Integrate all CVPR risk-closure evidence into ledger v12."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_cvpr_evidence_v12.json",
    )
    return parser.parse_args()


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _objects_with_key(value, key: str) -> list[dict]:
    rows = []
    if isinstance(value, dict):
        if key in value:
            rows.append(value)
        for child in value.values():
            rows.extend(_objects_with_key(child, key))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_objects_with_key(child, key))
    return rows


def main() -> None:
    args = parse_args()
    evidence = copy.deepcopy(_read("outputs/reports/fp_cvpr_evidence_v11.json"))
    final = _read("outputs/reports/fevp_final_claim_v4.json")
    closure = _read("outputs/reports/cvpr_risk_closure_v1.json")
    stability = _read("outputs/reports/fevp_fiveseed_stability_v2.json")
    graph = _read("outputs/reports/fevp_graph_metrics_v2.json")
    prompts = _read("outputs/reports/fevp_adversarial_prompts_v3.json")
    budget = _read("outputs/reports/fevp_budget_matched_control_v2.json")
    attribution = _read("outputs/reports/fevp_seed_attribution_v2.json")
    deletion = _read("outputs/reports/fevp_room_deletion_diagnosis_v2.json")
    external = _read("outputs/reports/external_baseline_protocol_audit_v2.json")
    attacks = _read("outputs/reports/fevp_attack_supplement_v1.json")
    claim_audit = _read("outputs/reports/cvpr_claim_text_audit_v1.json")
    effect_rows = _objects_with_key(graph, "rank_biserial")
    effect_rows.extend(_objects_with_key(prompts, "rank_biserial"))
    category_summary = {
        name: {
            "cases": row["cases"],
            "typed_outside_exact": row["arms"]["typed_edit_graph"][
                "outside_pixel_exact"
            ],
            "typed_violation": row["arms"]["typed_edit_graph"][
                "locality_violation_rate"
            ],
            "no_gate_violation": row["arms"]["no_edit_gating"][
                "locality_violation_rate"
            ],
        }
        for name, row in prompts["categories"].items()
    }
    statistics = {
        "five_seed_ci_method": stability["metrics"]["delta_iou"]["ci_method"],
        "five_seed_degrees_of_freedom": stability["metrics"]["delta_iou"][
            "ci_degrees_of_freedom"
        ],
        "metric_seed_ranges": {
            name: row["seed_range"] for name, row in stability["metrics"].items()
        },
        "wilcoxon_effect_size_objects": len(effect_rows),
        "wilcoxon_effect_sizes_complete": all(
            {
                "rank_biserial",
                "cliffs_delta",
                "oriented_rank_biserial",
                "oriented_cliffs_delta",
            }
            <= set(row)
            for row in effect_rows
        ),
        "prompt_categories": category_summary,
        "canonical_text_audit": claim_audit,
    }
    causal = {
        "budget_matched": budget,
        "seed_attribution": attribution,
        "room_deletion": deletion,
    }
    protocol = {
        "formal_spec": "docs/fevp_formal_spec_v3.md",
        "attack_supplement": attacks,
        "validity_watertight": (
            "outputs/reports/fevp_validity_watertight_v2.json"
        ),
        "metric_contract": "configs/fevp_metric_contract_v2.json",
    }
    reproducibility = {
        "lockfiles": [
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "configs/fp_raster2seq_requirements.txt",
        ],
        "provenance_builder": "scripts/build_cvpr_artifact_provenance_v1.py",
        "provenance_auditor": "scripts/audit_cvpr_artifact_provenance_v1.py",
        "deterministic_archive_module": (
            "src/floorflow/evaluation/deterministic_archive.py"
        ),
        "deterministic_archive_contract": closure[
            "deterministic_archive_contract"
        ],
    }
    new_gates = {
        **closure["checks"],
        "all_24_numbered_checks_resolved": (
            closure["resolved"] == closure["items"]
        ),
        "wilcoxon_effect_sizes_complete": statistics[
            "wilcoxon_effect_sizes_complete"
        ],
        "prompt_categories_complete": len(category_summary) == 4,
        "external_protocol_audit_complete": all(external["checks"].values()),
        "formal_attack_coverage_complete": all(attacks["checks"].values()),
        "canonical_claim_audit_complete": claim_audit["status"] == "complete",
    }
    evidence["claims"]["supported_protocol_v2"] = [
        (
            "The typed edit graph preserves outside exactness 1.0 and improves "
            "Delta-IoU and graph consistency over dense and graphless routing; "
            "the watertightness point-estimate trade-off includes a five-seed CI "
            "in fevp_fiveseed_stability_v2.json."
        )
    ]
    failed = [name for name, passed in new_gates.items() if not passed]
    evidence.update(
        {
            "status": "complete" if not failed else "failed",
            "claim_status": "cvpr_risk_closure_ready",
            "primary_narrative": final["claim_en"],
            "release_boundary": (
                "FEVP protocol and strict-locality evidence only. No HouseMind "
                "defeat, generation SOTA, human preference, or watertight-guard "
                "benefit claim."
            ),
            "submission_status": "cvpr_experiment_evidence_ready_v17",
            "experiment_objective_complete": not failed,
            "package_target": "outputs/cvpr_package_fp_v17",
            "risk_closure": {
                "requested_risk_themes": 22,
                "numbered_checks": closure["items"],
                "resolved_checks": closure["resolved"],
                "source_report": "outputs/reports/cvpr_risk_closure_v1.json",
            },
            "statistical_governance": statistics,
            "causal_mechanisms": causal,
            "external_protocol": external,
            "protocol_hardening": protocol,
            "reproducibility": reproducibility,
            "canonical_narrative": {
                "claim_sections": "paper/cvpr_claim_sections_v1.md",
                "claim_audit": "outputs/reports/cvpr_claim_text_audit_v1.json",
                "house_mind": external["housemind"],
                "ce2eplan": external["ce2eplan"],
            },
        }
    )
    evidence["gates"].update(new_gates)
    evidence["new_gate_checks"] = new_gates
    evidence["protocol_sources"].update(
        {
            "risk_closure": "outputs/reports/cvpr_risk_closure_v1.json",
            "statistics": "outputs/reports/fevp_fiveseed_stability_v2.json",
            "causal_budget_match": (
                "outputs/reports/fevp_budget_matched_control_v2.json"
            ),
            "external_protocol": (
                "outputs/reports/external_baseline_protocol_audit_v2.json"
            ),
            "formal_spec": "docs/fevp_formal_spec_v3.md",
            "claim_audit": "outputs/reports/cvpr_claim_text_audit_v1.json",
        }
    )
    evidence["claims"]["supported_risk_closure"] = [
        (
            "All 24 numbered checks in the supplied CVPR self-audit resolve to "
            "machine-readable evidence."
        ),
        (
            "Five-seed intervals use Student-t across seeds; paired case contrasts "
            "declare paired bootstrap iterations and seeds."
        ),
        (
            "Every reported Wilcoxon comparison includes rank-biserial and Cliff's "
            "delta effect sizes."
        ),
        (
            "Raster2Seq matched rows use shared project preprocessing and are not "
            "described as official preprocessing."
        ),
    ]
    if failed:
        raise ValueError(f"risk-closure evidence gates failed: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "risk_checks": closure["items"],
                "gates": len(new_gates),
                "failed": failed,
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()

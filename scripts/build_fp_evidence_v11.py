#!/usr/bin/env python3
"""Integrate FEVP portability and adversarial prompt robustness evidence."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_cvpr_evidence_v11.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = copy.deepcopy(_read("outputs/reports/fp_cvpr_evidence_v10.json"))
    final = _read("outputs/reports/fevp_final_claim_v4.json")
    core = _read("outputs/reports/fevp_core_audit_coverage_v3.json")
    ce2e = _read("outputs/reports/fevp_ce2eplan_portability_v3.json")
    house_gate = _read("outputs/reports/fevp_housemind_portability_gate_v1.json")
    prompts = _read("outputs/reports/fevp_adversarial_prompts_v3.json")
    evidence.update(
        {
            "status": "complete",
            "claim_status": "portable_fevp_and_prompt_robustness_ready",
            "primary_narrative": final["claim_en"],
            "released_claim": final["claim_en"],
            "portability_and_robustness": {
                "final_claim": final,
                "core_audit": core,
                "ce2eplan": ce2e,
                "house_mind_gate": house_gate,
                "adversarial_prompts": prompts,
            },
            "submission_status": (
                "cvpr_experiment_evidence_ready_for_portable_fevp_claim"
            ),
            "experiment_objective_complete": True,
            "package_target": "outputs/cvpr_package_fp_v16",
        }
    )
    evidence["claims"]["supported_portability_and_robustness"] = [
        final["portability_claim"],
        final["prompt_robustness_claim"],
        (
            f"HouseMind portability remains an explicit external gate: "
            f"{house_gate['official_code_status']}."
        ),
        (
            "The core audit table reports HouseMind and CE2EPlan locality cells "
            "as not reported, not as failures."
        ),
    ]
    evidence["claims"]["not_supported"] = list(
        dict.fromkeys(
            evidence["claims"]["not_supported"] + final["forbidden_claims"]
        )
    )
    new_gates = {
        **final["checks"],
        "core_audit_missing_not_failure": (
            "does not mean" in core["footnote"]
        ),
        "ce2eplan_portable_metrics_finite": ce2e["checks"][
            "portable_validity_and_graph_metrics"
        ],
        "house_mind_gate_blocked": house_gate["status"] == "blocked_external",
        "prompt_proxy_four_categories": prompts["checks"][
            "four_categories_complete"
        ],
        "prompt_proxy_no_gate_leakage": prompts["checks"][
            "no_gate_exposes_leakage"
        ],
    }
    evidence["gates"].update(new_gates)
    evidence["new_gate_checks"] = new_gates
    evidence["protocol_sources"].update(
        {
            "final_claim_v4": "outputs/reports/fevp_final_claim_v4.json",
            "core_audit": "outputs/reports/fevp_core_audit_coverage_v3.json",
            "ce2eplan_portability": (
                "outputs/reports/fevp_ce2eplan_portability_v3.json"
            ),
            "housemind_gate": (
                "outputs/reports/fevp_housemind_portability_gate_v1.json"
            ),
            "adversarial_prompts": (
                "outputs/reports/fevp_adversarial_prompts_v3.json"
            ),
        }
    )
    failures = [key for key, value in new_gates.items() if not value]
    if failures:
        raise ValueError(f"floorplan v11 gates failed: {failures}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": evidence["status"], "new_gates": len(new_gates), "failed": failures},
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()

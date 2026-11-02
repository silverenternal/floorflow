#!/usr/bin/env python3
"""Integrate locality-auditability reframing and public audit maps."""

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
        default=ROOT / "outputs/reports/fp_cvpr_evidence_v10.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = copy.deepcopy(_read("outputs/reports/fp_cvpr_evidence_v9.json"))
    final = _read("outputs/reports/fevp_final_claim_v3.json")
    house = _read("outputs/reports/fevp_housemind_positioning_v2.json")
    pareto = _read("outputs/reports/fevp_mechanism_pareto_v3.json")
    audit_map = _read("outputs/reports/fevp_audit_evidence_map_v2.json")
    deletion = _read("outputs/reports/fevp_room_deletion_diagnosis_v2.json")
    evidence.update(
        {
            "status": "complete",
            "claim_status": "locality_auditability_gap_claim_ready",
            "primary_narrative": final["claim_en"],
            "released_claim": final["claim_en"],
            "problem_statement": final["problem_statement"],
            "locality_auditability_evidence": {
                "final_claim": final,
                "house_mind": house,
                "pareto_v3": pareto,
                "audit_map": audit_map,
                "room_deletion": deletion,
            },
            "submission_status": (
                "cvpr_experiment_evidence_ready_for_auditability_gap_claim"
            ),
            "experiment_objective_complete": True,
            "package_target": "outputs/cvpr_package_fp_v15",
        }
    )
    evidence["claims"]["supported_auditability_gap_claim"] = [
        final["claim_en"],
        (
            "HouseMind-E/O are explicitly acknowledged as multimodal instruction "
            "editing context; their public Table 3 lacks FEVP locality audit evidence."
        ),
        (
            "HouseMind-G is generation-only and is excluded from editing Pareto "
            "and editing rankings."
        ),
        (
            "The audit evidence map marks missing cells with a dash and states that "
            "they mean not reported, not failed."
        ),
        (
            "Room deletion is the top-decile GED difficulty mode while strict "
            "outside exactness remains 1.0 and violation rate remains zero."
        ),
    ]
    evidence["claims"]["not_supported"] = list(
        dict.fromkeys(
            evidence["claims"]["not_supported"] + final["forbidden_claims"]
        )
    )
    new_gates = {
        **final["checks"],
        "audit_map_complete": audit_map["checks"]["floorflow_complete"],
        "housemind_missing_cells_explained": (
            "does not mean" in audit_map["footnote"]
        ),
        "house_mind_generation_only_excluded": house["checks"][
            "generation_variant_not_ranked"
        ],
        "room_deletion_tail_share_dominant": deletion["checks"][
            "room_deletion_dominates_tail"
        ],
    }
    evidence["gates"].update(new_gates)
    evidence["new_gate_checks"] = new_gates
    evidence["protocol_sources"].update(
        {
            "final_claim_v3": "outputs/reports/fevp_final_claim_v3.json",
            "house_mind_positioning": (
                "outputs/reports/fevp_housemind_positioning_v2.json"
            ),
            "mechanism_pareto_v3": (
                "outputs/reports/fevp_mechanism_pareto_v3.json"
            ),
            "audit_evidence_map": (
                "outputs/reports/fevp_audit_evidence_map_v2.json"
            ),
            "room_deletion_diagnosis": (
                "outputs/reports/fevp_room_deletion_diagnosis_v2.json"
            ),
        }
    )
    failures = [key for key, value in new_gates.items() if not value]
    if failures:
        raise ValueError(f"floorplan v10 gates failed: {failures}")
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

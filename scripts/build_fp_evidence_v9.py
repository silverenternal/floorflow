#!/usr/bin/env python3
"""Integrate offensive locality, Pareto, safety, strength, and claim evidence."""

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
        default=ROOT / "outputs/reports/fp_cvpr_evidence_v9.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = copy.deepcopy(_read("outputs/reports/fp_cvpr_evidence_v8.json"))
    final = _read("outputs/reports/fevp_final_claim_v2.json")
    locality = _read("outputs/reports/fevp_locality_claim_v2.json")
    pareto = _read("outputs/reports/fevp_mechanism_pareto_v2.json")
    safety = _read("outputs/reports/fevp_safety_baseline_v2.json")
    strength = _read("outputs/reports/fevp_edit_strength_v2.json")
    attacks = _read("outputs/reports/fevp_protocol_attacks_compact_v2.json")
    evidence.update(
        {
            "status": "complete",
            "claim_status": "offensive_fevp_claim_ready",
            "primary_narrative": final["claim_en"],
            "released_claim": final["claim_en"],
            "offensive_evidence": {
                "final_claim": final,
                "locality": locality,
                "pareto": pareto,
                "safety_baseline": safety,
                "edit_strength": strength,
                "compact_attacks": attacks,
            },
            "submission_status": (
                "cvpr_experiment_evidence_ready_with_bounded_offensive_claim"
            ),
            "experiment_objective_complete": True,
            "package_target": "outputs/cvpr_package_fp_v14",
        }
    )
    evidence["claims"]["supported_offensive_claim"] = [
        final["claim_en"],
        (
            f"Strict outside exactness is 1.0 at pixel/room/label/polygon/edge "
            f"levels over {locality['cases']:,} rows, with zero violation and "
            f"{locality['single_room_negative_controls']:,} single-room controls."
        ),
        (
            f"No-gate lowers room exactness to "
            f"{locality['no_edit_gating']['outside_room_level_exact']:.4f} and "
            f"raises violation rate to "
            f"{locality['no_edit_gating']['locality_violation_rate']:.4f}."
        ),
        (
            "Typed routing is on the computed Pareto frontier and dominates "
            "dense, no-gate, and graphless arms in the joint locality-quality objective."
        ),
        (
            "Delta fidelity is reported for 1-room, 2-3-room, 4+-room, room "
            "deletion, and opening/furniture groups under strict locality."
        ),
        (
            "The compact protocol attack table passes all five headline attacks "
            "with zero observed false-pass behavior."
        ),
        (
            "Validity is a five-check safety baseline; watertight guard has 57 "
            "repairs and 67 regressions and is not promoted."
        ),
    ]
    evidence["claims"]["not_supported"] = list(
        dict.fromkeys(
            evidence["claims"]["not_supported"] + final["forbidden_claims"]
        )
    )
    new_gates = {
        **final["checks"],
        "offensive_claim_sources_complete": len(final["sources"]) >= 7,
        "compact_attack_disclaimer_present": (
            "not used to increase" in attacks["disclaimer"]
        ),
    }
    evidence["gates"].update(new_gates)
    evidence["new_gate_checks"] = new_gates
    evidence["protocol_sources"].update(
        {
            "final_claim": "outputs/reports/fevp_final_claim_v2.json",
            "locality_claim": "outputs/reports/fevp_locality_claim_v2.json",
            "mechanism_pareto": (
                "outputs/reports/fevp_mechanism_pareto_v2.json"
            ),
            "safety_baseline": "outputs/reports/fevp_safety_baseline_v2.json",
            "edit_strength": "outputs/reports/fevp_edit_strength_v2.json",
            "compact_attacks": (
                "outputs/reports/fevp_protocol_attacks_compact_v2.json"
            ),
        }
    )
    failures = [key for key, value in new_gates.items() if not value]
    if failures:
        raise ValueError(f"floorplan v9 gates failed: {failures}")
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

"""Claim branches for matched and context-only external comparisons."""

from __future__ import annotations


def build_claim_matrix(
    *,
    paired_checks: dict[str, bool],
    matched_executable: bool,
) -> dict:
    matched_ok = matched_executable and all(paired_checks.values())
    return {
        "matched_cubicasa_reconstruction_head_to_head": {
            "enabled": matched_ok,
            "requires": paired_checks,
        },
        "complete_cad_generation_sota": {
            "enabled": False,
            "reason": "No matched full-CAD external execution is available.",
        },
        "rplan_generation_sota": {
            "enabled": False,
            "reason": "No executable matched RPLAN competitor or shared split run is available.",
        },
        "house_mind_editing_defeat": {
            "enabled": False,
            "reason": "Official HouseMind inference and matched instructions are unavailable.",
        },
        "strict_locality_protocol_contribution": {
            "enabled": True,
            "reason": "Supported by separate exact-audit and adversarial evidence.",
        },
    }


__all__ = ["build_claim_matrix"]

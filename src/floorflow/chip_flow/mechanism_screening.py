"""Shared promotion states for chip mechanism screens."""

from __future__ import annotations


def infer_stage(setting: str) -> str:
    if setting == "nfe4_refine400":
        return "quick"
    if setting == "nfe4_refine1600":
        return "confirmation"
    raise ValueError("stage is required for a non-standard screening setting")


def screen_status(stage: str, passed: bool) -> str:
    outcomes = {
        "quick": ("quick_pass_promote_to_confirmation", "rejected_quick_screen"),
        "confirmation": (
            "confirmation_pass_promote_to_full",
            "rejected_confirmation",
        ),
        "full": ("full_pass_promote_to_multiseed", "rejected_full_data"),
    }
    if stage not in outcomes:
        raise ValueError(f"unknown screening stage: {stage}")
    return outcomes[stage][0 if passed else 1]


def claim_boundary(stage: str) -> str:
    boundaries = {
        "quick": "Quick-screen evidence only; not a candidate.",
        "confirmation": (
            "Frozen-subset confirmation; not full-data or publication evidence."
        ),
        "full": (
            "Single-seed matched full-data check; multiseed and publication "
            "claims remain pending."
        ),
    }
    if stage not in boundaries:
        raise ValueError(f"unknown screening stage: {stage}")
    return boundaries[stage]

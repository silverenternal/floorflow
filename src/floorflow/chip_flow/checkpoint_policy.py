"""Frozen final-versus-best checkpoint policy for chip training reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch


def resolve_checkpoint_state(
    policy: str,
    *,
    final_state: Mapping[str, torch.Tensor],
    final_step: int,
    final_path: Path,
    best_path: Path,
    state_key: str,
    selection_report: Path | None = None,
) -> tuple[dict[str, torch.Tensor], int, Path, str]:
    """Return the state used for final evidence and its declared policy."""

    if policy == "final":
        return (
            dict(final_state),
            int(final_step),
            Path(final_path),
            f"last_step_{state_key}",
        )
    if policy == "legalized_best":
        if selection_report is None or not selection_report.is_file():
            raise FileNotFoundError(
                "legalized_best policy requires a checkpoint selection report"
            )
        report = json.loads(selection_report.read_text(encoding="utf-8"))
        selected = report.get("selected")
        if not isinstance(selected, dict) or "checkpoint" not in selected:
            raise ValueError("checkpoint selection report lacks a selected checkpoint")
        selected_path = Path(selected["checkpoint"])
        policy_name = "legalized_shortlist_validation"
    elif policy == "best":
        selected_path = best_path
        policy_name = f"best_validation_{state_key}"
    else:
        raise ValueError(f"unknown checkpoint policy: {policy}")
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"checkpoint policy target is missing: {selected_path}"
        )
    payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    if state_key not in payload:
        raise ValueError(f"best checkpoint lacks {state_key} state")
    return (
        dict(payload[state_key]),
        int(payload["step"]),
        selected_path,
        policy_name,
    )


__all__ = ["resolve_checkpoint_state"]

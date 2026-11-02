"""Stage-specific CAD channel readout with optional dynamic spatial gates."""

from __future__ import annotations

import torch
from torch import nn


def apply_cad_stage(
    name: str,
    class_ids: torch.Tensor,
    stage_features: torch.Tensor,
    stage_token: torch.Tensor,
    stage_gate: torch.Tensor,
    stage_scale: nn.Parameter,
    prior: torch.Tensor,
    *,
    dynamic_refiner: nn.Module | None,
    legacy_trunk: nn.Module | None,
    legacy_film: nn.Module | None,
    legacy_head: nn.Module | None,
    preserve_room_stage: bool,
    condition: torch.Tensor | None = None,
    base_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Apply one stage head without changing the caller's channel contract."""

    prior_slice = prior[:, class_ids]
    stage_prior = prior_slice if base_logits is None else base_logits
    spatial_gate = None
    if dynamic_refiner is not None:
        refined = dynamic_refiner(
            stage_features,
            stage_prior,
            stage_token,
            condition,
        )
        stage_residual = (
            refined["residual"] * refined["spatial_gate"][:, None]
        )
        spatial_gate = refined["spatial_gate"]
    else:
        if legacy_trunk is None or legacy_film is None or legacy_head is None:
            raise ValueError("legacy stage modules are incomplete")
        hidden = legacy_trunk(stage_features)
        scale, bias = legacy_film(stage_token).chunk(2, dim=-1)
        hidden = (
            hidden * (1.0 + scale[..., None, None])
            + bias[..., None, None]
        )
        stage_residual = legacy_head(hidden)
    scale = torch.tanh(stage_scale) * stage_gate[:, None, None, None]
    candidate_logits = (
        stage_prior + scale.to(stage_residual.dtype) * stage_residual
    )
    # Preserving the room stage means suppressing its direct replacement, not
    # discarding the typed channel head that produced ``stage_prior``.
    logits = (
        stage_prior
        if name == "rooms" and preserve_room_stage
        else candidate_logits
    )
    result = {
        "logits": logits,
        "candidate_logits": candidate_logits,
        "spatial_gate": spatial_gate,
        "residual": stage_residual,
    }
    if dynamic_refiner is not None:
        result["route_diagnostics"] = refined.get("route_diagnostics")
    return result


__all__ = ["apply_cad_stage"]

"""Initialize CFM sampling from contour-conditioned pixel distributions."""

from __future__ import annotations

import torch


def _unwrapped_flow_model(model):
    return getattr(model, "base", model)


@torch.no_grad()
def distribution_initial_state(
    model,
    distribution_head,
    footprint: torch.Tensor,
    floor_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Encode contour-only features and project foreground probabilities to state."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    base = _unwrapped_flow_model(model)
    zero = torch.zeros(
        footprint.shape[0],
        base.target_channels,
        *footprint.shape[-2:],
        device=footprint.device,
        dtype=footprint.dtype,
    )
    time = torch.zeros(footprint.shape[0], device=footprint.device)
    features = base._encode_features(
        zero,
        time,
        footprint,
        floor_ids=floor_ids,
    )
    output = distribution_head(features)
    foreground = output["probability"][:, 1:]
    if foreground.shape[1] != base.target_channels:
        raise ValueError(
            "distribution foreground channels must match target_channels"
        )
    state = foreground * footprint.float()
    return {
        "state": state,
        "probability": output["probability"],
        "reliability": output["reliability"],
    }


def blend_distribution_state(
    control: torch.Tensor,
    distribution: torch.Tensor,
    footprint: torch.Tensor,
    *,
    blend: float,
) -> torch.Tensor:
    """Blend predicted controls and pixel probabilities inside the footprint."""

    if not 0.0 <= float(blend) <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    state = (1.0 - float(blend)) * control + float(blend) * distribution
    return state * footprint.float()


__all__ = ["blend_distribution_state", "distribution_initial_state"]

"""Target-free sparse priors derived from contour distribution states."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def low_confidence_prior_mask(
    state: torch.Tensor,
    footprint: torch.Tensor,
    *,
    hidden_fraction: float = 0.25,
) -> torch.Tensor:
    """Mark the least-confident footprint pixels as unknown control."""

    if state.ndim != 4:
        raise ValueError("state must be [B,C,H,W]")
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if not 0.0 <= hidden_fraction <= 1.0:
        raise ValueError("hidden_fraction must be in [0,1]")
    inside = F.interpolate(
        footprint.float(),
        size=state.shape[-2:],
        mode="nearest",
    ).bool()
    confidence = state.float().amax(dim=1)
    hidden = torch.zeros_like(inside)
    for index in range(state.shape[0]):
        valid = inside[index, 0]
        count = int(valid.sum())
        if count == 0 or hidden_fraction <= 0.0:
            continue
        keep = min(
            count,
            max(1, round(count * float(hidden_fraction))),
        )
        values = confidence[index][valid]
        threshold = values.kthvalue(min(keep, values.numel())).values
        hidden[index, 0] = (confidence[index] <= threshold) & valid
    return hidden


__all__ = ["low_confidence_prior_mask"]

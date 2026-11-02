"""Spatial room-query utility functions."""

from __future__ import annotations

import torch


def mask_centers(mask_logits: torch.Tensor) -> torch.Tensor:
    probability = mask_logits.sigmoid()
    height, width = probability.shape[-2:]
    y = torch.linspace(
        0.0,
        1.0,
        height,
        device=probability.device,
        dtype=probability.dtype,
    )
    x = torch.linspace(
        0.0,
        1.0,
        width,
        device=probability.device,
        dtype=probability.dtype,
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    mass = probability.flatten(2).sum(dim=-1).clamp_min(1.0e-5)
    center_x = (
        probability * grid_x[None, None]
    ).flatten(2).sum(dim=-1) / mass
    center_y = (
        probability * grid_y[None, None]
    ).flatten(2).sum(dim=-1) / mass
    return torch.stack(
        [center_x.clamp(0.0, 1.0), center_y.clamp(0.0, 1.0)],
        dim=-1,
    )


__all__ = ["mask_centers"]

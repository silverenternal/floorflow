"""Confidence-aware weighting for shared semantic channel residuals."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reliability_weighted_residual(
    residual: torch.Tensor,
    reliability: torch.Tensor,
    *,
    floor: float = 0.75,
) -> torch.Tensor:
    """Down-weight uncertain channel residuals without hard masking."""

    if not 0.0 < float(floor) <= 1.0:
        raise ValueError("reliability floor must be in (0,1]")
    if reliability.ndim == 3:
        reliability = reliability[:, None]
    if (
        reliability.ndim != 4
        or reliability.shape[0] != residual.shape[0]
        or reliability.shape[1] != 1
    ):
        raise ValueError("reliability must be [B,1,H,W] matching residual")
    if reliability.shape[-2:] != residual.shape[-2:]:
        reliability = F.interpolate(
            reliability.float(),
            size=residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    weight = float(floor) + (1.0 - float(floor)) * reliability.to(
        residual.dtype
    ).clamp(0.0, 1.0)
    return residual * weight


__all__ = ["reliability_weighted_residual"]

"""Bounded parent-support gating for hierarchical element masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def parent_support_band(
    parent: torch.Tensor,
    footprint: torch.Tensor,
    *,
    radius: int = 2,
) -> torch.Tensor:
    """Normalize parent confidence, dilate it, and keep support in footprint."""

    if parent.ndim == 3:
        parent = parent[:, None]
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    peak = parent.flatten(2).amax(dim=2).clamp_min(1.0e-6)
    relative = parent / peak[:, :, None, None]
    kernel = 2 * max(0, int(radius)) + 1
    band = F.max_pool2d(
        relative,
        kernel_size=kernel,
        stride=1,
        padding=int(radius),
    ).clamp(0.0, 1.0)
    if footprint.shape[-2:] != band.shape[-2:]:
        footprint = F.interpolate(
            footprint.float(),
            size=band.shape[-2:],
            mode="nearest",
        )
    return band * footprint.float()


def apply_support_gate(
    logits: torch.Tensor,
    parent: torch.Tensor,
    footprint: torch.Tensor,
    *,
    strength: float,
    radius: int = 2,
) -> torch.Tensor:
    """Suppress child logits only outside a dilated parent-support band."""

    if strength <= 0.0:
        return logits
    band = parent_support_band(parent, footprint, radius=radius)
    if band.shape[-2:] != logits.shape[-2:]:
        band = F.interpolate(
            band,
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return logits - float(strength) * (1.0 - band)


__all__ = ["apply_support_gate", "parent_support_band"]

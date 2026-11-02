"""Differentiable edge evidence induced by competing room-mask logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def partition_edge_evidence(
    mask_logits: torch.Tensor,
    *,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Return one high-resolution map where adjacent room supports differ."""

    if mask_logits.ndim != 4:
        raise ValueError("mask_logits must be [B,Q,H,W]")
    probability = mask_logits.sigmoid()
    horizontal = (
        probability[:, :, :, 1:] - probability[:, :, :, :-1]
    ).abs().sum(dim=1)
    vertical = (
        probability[:, :, 1:, :] - probability[:, :, :-1, :]
    ).abs().sum(dim=1)
    evidence = F.pad(horizontal, (0, 1)) + F.pad(vertical, (0, 0, 0, 1))
    return ((evidence - float(threshold)) / (1.0 - float(threshold))).clamp(
        0.0,
        1.0,
    )


__all__ = ["partition_edge_evidence"]

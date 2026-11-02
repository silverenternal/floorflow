"""Normalized balancing for heterogeneous chip training objectives."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def normalized_loss_balance(
    losses: Mapping[str, torch.Tensor],
    *,
    floor: float = 1.0e-6,
    max_weight: float = 1.0e4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equalize detached objective scales while preserving each gradient path."""

    if not losses:
        raise ValueError("loss balancing requires at least one objective")
    if floor <= 0.0 or max_weight <= 0.0:
        raise ValueError("loss-balance floor and maximum weight must be positive")
    names = tuple(losses)
    values = torch.stack([losses[name].float() for name in names])
    scales = values.detach().abs().clamp_min(float(floor))
    raw_weights = scales.reciprocal().clamp_max(float(max_weight))
    weights = raw_weights / raw_weights.mean().clamp_min(1.0e-12)
    total = (values * weights).sum() / len(names)
    return total, {
        f"loss_balance_{name}": weights[index].detach()
        for index, name in enumerate(names)
    }


__all__ = ["normalized_loss_balance"]

"""Differentiable room-matching costs shared by training heads."""

from __future__ import annotations

import torch


def pairwise_mask_cost(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Dice plus light BCE cost for query-mask assignment."""

    predicted = torch.nan_to_num(
        predicted.float(),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )
    target = torch.nan_to_num(
        target.float(),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    predicted_flat = predicted.flatten(1)
    target_flat = target.flatten(1)
    intersection = predicted_flat @ target_flat.T
    denominator = (
        predicted_flat.sum(1)[:, None]
        + target_flat.sum(1)[None, :]
        - intersection
    )
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    bce = -(
        torch.log(predicted_flat + 1.0e-6) @ target_flat.T
        + torch.log(1.0 - predicted_flat + 1.0e-6)
        @ (1.0 - target_flat).T
    ) / predicted_flat.shape[1]
    return dice + 0.2 * bce


__all__ = ["pairwise_mask_cost"]


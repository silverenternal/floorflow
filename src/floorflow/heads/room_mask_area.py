"""Scale-aware room-mask area supervision."""

from __future__ import annotations

import torch


def symmetric_area_error(
    predicted_probability: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Return a bounded per-mask relative area error.

    Dice is sensitive to overlap but can be dominated by translation and
    boundary terms in a multi-objective decoder. This term isolates scale:
    under- and over-segmentation are both mapped to ``[0, 1]`` while retaining
    a non-zero gradient at a collapsed probability map.
    """

    if predicted_probability.shape != target_mask.shape:
        raise ValueError("predicted and target masks must share a shape")
    dimensions = tuple(range(1, predicted_probability.ndim))
    predicted_area = predicted_probability.float().mean(dim=dimensions)
    target_area = target_mask.float().mean(dim=dimensions)
    denominator = predicted_area + target_area + float(epsilon)
    return (predicted_area - target_area).abs() / denominator


def matched_mask_area_loss(
    predicted_probability: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Average relative area error over already matched room queries."""

    return symmetric_area_error(
        predicted_probability,
        target_mask,
    ).mean()


__all__ = ["matched_mask_area_loss", "symmetric_area_error"]

"""Per-case binary objectives aligned with case-averaged graph metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def per_case_binary_cross_entropy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    weight: torch.Tensor | None = None,
    pos_weight: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Average BCE over cases rather than over pooled graph edges."""

    prediction, target, mask, weight = _flatten_inputs(
        prediction,
        target,
        mask,
        weight,
    )
    if (
        pos_weight is not None
        and torch.is_tensor(pos_weight)
        and pos_weight.ndim > 0
    ):
        pos_weight = pos_weight.reshape(prediction.shape[0], -1)
        if pos_weight.shape[1] != 1:
            raise ValueError("pos_weight must have one value per case")
    logits = torch.logit(prediction.float().clamp(1.0e-6, 1.0 - 1.0e-6))
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )
    sample_weight = mask.float()
    if weight is not None:
        sample_weight = sample_weight * weight.float()
    denominator = sample_weight.sum(dim=1)
    active = denominator > 0.0
    if not bool(active.any()):
        return prediction.sum() * 0.0
    per_case = (
        (elementwise * sample_weight).sum(dim=1)
        / denominator.clamp_min(1.0e-6)
    )
    return per_case[active].mean()


def per_case_soft_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return case-averaged soft Dice loss for sparse graph tensors."""

    prediction, target, mask, _ = _flatten_inputs(
        prediction,
        target,
        mask,
        None,
    )
    prediction = prediction.float().clamp(0.0, 1.0) * mask
    target = target.float() * mask
    intersection = (prediction * target).sum(dim=1)
    denominator = prediction.sum(dim=1) + target.sum(dim=1)
    active = mask.flatten(1).any(dim=1)
    if not bool(active.any()):
        return prediction.sum() * 0.0
    loss = 1.0 - (2.0 * intersection + 1.0) / (
        denominator + 1.0
    )
    return loss[active].mean()


def per_case_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Return case-averaged focal loss over masked pixels."""

    logits, target, mask, _ = _flatten_inputs(
        logits,
        target,
        mask,
        None,
    )
    probability = logits.sigmoid()
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    probability_target = (
        probability * target
        + (1.0 - probability) * (1.0 - target)
    )
    alpha_target = alpha * target + (1.0 - alpha) * (1.0 - target)
    elementwise = (
        alpha_target
        * (1.0 - probability_target).pow(gamma)
        * elementwise
    )
    denominator = mask.float().sum(dim=1)
    active = denominator > 0.0
    if not bool(active.any()):
        return elementwise.sum() * 0.0
    per_case = (
        (elementwise * mask.float()).sum(dim=1)
        / denominator.clamp_min(1.0)
    )
    return per_case[active].mean()


def _flatten_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share a shape")
    batch = prediction.shape[0]
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    mask = (
        torch.ones_like(prediction, dtype=torch.bool)
        if mask is None
        else mask.bool().flatten(1)
    )
    if mask.shape != prediction.shape:
        raise ValueError("mask must match prediction shape")
    if weight is not None:
        weight = weight.float().flatten(1)
        if weight.shape != prediction.shape:
            raise ValueError("weight must match prediction shape")
    if prediction.shape[0] != batch:
        raise RuntimeError("flattening changed the batch dimension")
    return prediction, target, mask, weight


__all__ = [
    "per_case_binary_cross_entropy",
    "per_case_focal_loss",
    "per_case_soft_dice",
]

"""Room-instance boundary supervision for owner refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def room_boundary_targets(
    masks: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if masks.ndim != 4 or valid.shape != masks.shape[:2]:
        raise ValueError("masks must be [B,Q,H,W] and valid [B,Q]")
    union = (masks * valid[:, :, None, None]).amax(dim=1)
    eroded = -F.max_pool2d(
        -union[:, None],
        kernel_size=3,
        stride=1,
        padding=1,
    )[:, 0]
    boundary = (union - eroded).clamp(0.0, 1.0)
    return boundary, union


def room_boundary_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    *,
    positive_weight: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.ndim == 4:
        if logits.shape[1] != 1:
            raise ValueError("boundary logits must have one channel")
        logits = logits[:, 0]
    if logits.ndim != 3:
        raise ValueError("boundary logits must be [B,H,W]")
    boundary, union = room_boundary_targets(masks, valid)
    boundary = F.interpolate(
        boundary[:, None],
        size=logits.shape[-2:],
        mode="nearest",
    )[:, 0]
    union = F.interpolate(
        union[:, None],
        size=logits.shape[-2:],
        mode="nearest",
    )[:, 0]
    losses = F.binary_cross_entropy_with_logits(
        logits,
        boundary,
        reduction="none",
        pos_weight=logits.new_tensor(float(positive_weight)),
    )
    weights = union
    loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, {
        "room_boundary_loss": loss.detach(),
        "room_boundary_positive_rate": (
            (boundary * weights).sum() / weights.sum().clamp_min(1.0)
        ).detach(),
    }


__all__ = ["room_boundary_loss", "room_boundary_targets"]

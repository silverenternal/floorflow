"""Losses and targets for joint room-structure control repair."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize


def room_index_target(
    labels: torch.Tensor,
    room_ids: torch.Tensor,
) -> torch.Tensor:
    lookup = torch.full(
        (max(int(labels.max()), int(room_ids.max())) + 1,),
        -100,
        dtype=torch.long,
        device=labels.device,
    )
    lookup[room_ids] = torch.arange(len(room_ids), device=labels.device)
    safe_labels = labels.clamp_max(len(lookup) - 1)
    return lookup[safe_labels]


def masked_room_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if valid.ndim == 3:
        valid = valid[:, None]
    valid_pixels = (target >= 0) & valid[:, 0].bool()
    if not valid_pixels.any():
        zero = logits.sum() * 0.0
        return zero, {"room_ce": zero.detach(), "room_dice": zero.detach()}
    ce = F.cross_entropy(
        logits,
        target.clamp_min(0),
        reduction="none",
    )
    ce_loss = ce[valid_pixels].mean()
    probability = logits.softmax(dim=1) * valid_pixels[:, None]
    one_hot = F.one_hot(
        target.clamp_min(0),
        logits.shape[1],
    ).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid_pixels[:, None]
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = (
        probability.sum(dim=(0, 2, 3))
        + one_hot.sum(dim=(0, 2, 3))
    )
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return ce_loss + dice, {
        "room_ce": ce_loss.detach(),
        "room_dice": dice.detach(),
    }


def masked_focal_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    valid = valid.bool()
    if not valid.any():
        return logits.sum() * 0.0
    probability = logits.sigmoid().clamp(1.0e-4, 1.0 - 1.0e-4)
    positive = target * (1.0 - probability).pow(2.0) * probability.log()
    negative = (
        (1.0 - target).pow(4.0)
        * probability.pow(2.0)
        * (1.0 - probability).log()
    )
    focal = -((positive + negative) * valid).sum() / valid.sum().clamp_min(1)
    dice_probability = logits.sigmoid() * valid
    dice_target = target * valid
    intersection = (dice_probability * dice_target).sum(dim=(2, 3))
    denominator = (
        dice_probability.sum(dim=(2, 3))
        + dice_target.sum(dim=(2, 3))
    )
    dice = 1.0 - (
        (2.0 * intersection + 1.0) / (denominator + 1.0)
    ).mean()
    return focal + dice


def wall_centerline_target(wall: torch.Tensor) -> torch.Tensor:
    """Extract a one-pixel wall-centerline target from wall bands."""

    if wall.ndim == 4:
        wall = wall[:, 0]
    rows = [
        skeletonize(row > 0.5).astype(np.float32)
        for row in wall.detach().cpu().numpy()
    ]
    return torch.from_numpy(np.stack(rows, axis=0)[:, None]).to(
        wall.device
    )


def structure_losses(
    output: dict[str, torch.Tensor],
    structure_target: torch.Tensor,
    missing: torch.Tensor,
    room_count: int,
    opening_ids: torch.Tensor,
    centerline_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    missing = missing[:, None].bool()
    wall_target = structure_target[:, room_count : room_count + 1]
    opening_target = structure_target[
        :, room_count + 1 : room_count + 2
    ]
    wall_loss = masked_focal_dice(
        output["wall_logits"],
        wall_target,
        missing,
    )
    opening_loss = masked_focal_dice(
        output["opening_logits"],
        opening_target,
        missing,
    )
    lookup = torch.full(
        (int(opening_ids.max()) + 1,),
        -100,
        dtype=torch.long,
        device=opening_ids.device,
    )
    lookup[opening_ids] = torch.arange(len(opening_ids), device=opening_ids.device)
    local_type = lookup[
        structure_target[:, room_count + 2 : room_count + 2 + len(opening_ids)]
        .argmax(dim=1)
    ]
    type_valid = (local_type >= 0) & missing[:, 0]
    if type_valid.any():
        type_loss = F.cross_entropy(
            output["opening_type_logits"].permute(0, 2, 3, 1)[type_valid],
            local_type[type_valid],
        )
    else:
        type_loss = output["opening_logits"].sum() * 0.0
    centerline_loss = output["wall_logits"].sum() * 0.0
    if (
        float(centerline_weight) > 0.0
        and "centerline_logits" in output
    ):
        centerline_loss = masked_focal_dice(
            output["centerline_logits"],
            wall_centerline_target(wall_target),
            missing,
        )
    total = (
        wall_loss
        + 2.0 * opening_loss
        + 0.5 * type_loss
        + float(centerline_weight) * centerline_loss
    )
    return total, {
        "wall": wall_loss.detach(),
        "opening": opening_loss.detach(),
        "opening_type": type_loss.detach(),
        "wall_centerline": centerline_loss.detach(),
    }


__all__ = [
    "masked_focal_dice",
    "masked_room_loss",
    "room_index_target",
    "structure_losses",
    "wall_centerline_target",
]

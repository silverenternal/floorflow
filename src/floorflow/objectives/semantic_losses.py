"""Boundary-aware losses for room and wall maps."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from floorflow.data.taxonomy import FloorplanTaxonomy


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def boundary_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    input_tensor = mask[:, None] if mask.ndim == 3 else mask
    eroded = -F.max_pool2d(-input_tensor, 3, stride=1, padding=1)
    if mask.ndim == 3:
        eroded = eroded[:, 0]
    return (mask - eroded).clamp(0.0, 1.0)


def boundary_focal_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    boundary = boundary_mask(target)
    boundary_logits = logits * (1.0 + 3.0 * boundary)
    return focal_loss(boundary_logits, target) + dice_loss(logits, target)


def boundary_distance_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Soft Hausdorff-style weighting around target class transitions."""

    valid = target >= 0
    if not valid.any():
        return logits.sum() * 0.0
    target_np = target.detach().cpu().numpy()
    weights = np.zeros(target_np.shape, dtype=np.float32)
    for index, drawing in enumerate(target_np):
        boundary = np.zeros_like(drawing, dtype=bool)
        valid_drawing = drawing >= 0
        boundary[:, 1:] |= (
            valid_drawing[:, 1:]
            & valid_drawing[:, :-1]
            & (drawing[:, 1:] != drawing[:, :-1])
        )
        boundary[:, :-1] |= boundary[:, 1:]
        boundary[1:, :] |= (
            valid_drawing[1:, :]
            & valid_drawing[:-1, :]
            & (drawing[1:, :] != drawing[:-1, :])
        )
        boundary[:-1, :] |= boundary[1:, :]
        if boundary.any():
            distance = distance_transform_edt(~boundary)
            weights[index] = np.exp(-distance / max(1.0e-3, float(temperature)))
    distance_weight = torch.from_numpy(weights).to(
        device=logits.device,
        dtype=logits.dtype,
    )
    distance_weight = distance_weight * valid.float()
    ce = F.cross_entropy(
        logits,
        target.clamp_min(0),
        reduction="none",
    )
    return (ce * distance_weight).sum() / distance_weight.sum().clamp_min(1.0)


def wall_focal_dice(logits: torch.Tensor, wall_target: torch.Tensor) -> torch.Tensor:
    return focal_loss(logits, wall_target) + dice_loss(logits, wall_target)


def room_group_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-entropy and Dice restricted to visible room pixels."""

    room_ids = list(taxonomy.room_ids)
    room_logits = logits[:, [class_id - 1 for class_id in room_ids]]
    class_lookup = torch.full((taxonomy.num_classes,), -100, dtype=torch.long, device=target.device)
    class_lookup[torch.tensor(room_ids, device=target.device)] = torch.arange(len(room_ids), device=target.device)
    room_target = class_lookup[target]
    valid = room_target >= 0
    if not valid.any():
        zero = room_logits.sum() * 0.0
        return zero, zero.detach(), zero.detach()
    ce = F.cross_entropy(
        room_logits,
        room_target.clamp_min(0),
        weight=class_weights,
        reduction="none",
    )[valid].mean()
    probability = room_logits.softmax(dim=1)
    one_hot = F.one_hot(room_target.clamp_min(0), len(room_ids)).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid[:, None].float()
    valid_float = valid[:, None].float()
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = (probability * valid_float).sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_component = 1.0 - dice[present].mean()
    return ce + dice_component, ce.detach(), dice_component.detach()


@torch.no_grad()
def room_group_iou(
    logits: torch.Tensor,
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> float:
    room_ids = list(taxonomy.room_ids)
    prediction = torch.tensor(room_ids, device=logits.device)[
        logits[:, [class_id - 1 for class_id in room_ids]].argmax(dim=1)
    ]
    values = []
    for class_id in room_ids:
        expected = target == class_id
        if not expected.any():
            continue
        predicted = prediction == class_id
        values.append(
            float((predicted & expected).sum().float() / (predicted | expected).sum().clamp_min(1).float())
        )
    return float(np.mean(values)) if values else 0.0

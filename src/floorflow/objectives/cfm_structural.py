"""Structure-aware objectives on the one-step reconstruction of a CFM field."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.flow.masked_control import control_channel_groups


@dataclass(frozen=True)
class CFMStructuralWeights:
    family_bce: float = 0.10
    family_dice: float = 0.05
    opening_wall: float = 0.05
    wall_boundary: float = 0.05


def predicted_x1(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Recover the endpoint implied by the model's current velocity field."""

    return x_t + (1.0 - time[:, None, None, None]) * velocity


def _family_bce_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    channels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if channels.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, zero
    logits = logits[:, channels]
    target = target[:, channels]
    mask = footprint.expand_as(logits)
    positive = (target * mask).sum(dim=(0, 2, 3))
    negative = ((1.0 - target) * mask).sum(dim=(0, 2, 3))
    positive_weight = (negative / positive.clamp_min(1.0)).sqrt().clamp(1.0, 20.0)
    bce_pixels = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    bce = (
        bce_pixels * mask * positive_weight[None, :, None, None]
    ).sum() / (mask.expand_as(logits) * positive_weight[None, :, None, None]).sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0).mean()
    probability = torch.sigmoid(logits) * mask
    target = target * mask
    intersection = (probability * target).sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce, (1.0 - dice).mean()


def _channel(channel_id: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([int(channel_id)], device=device, dtype=torch.long)


def _wall_boundary_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    wall_id = taxonomy.class_id("wall") - 1
    wall_probability = torch.sigmoid(predicted[:, wall_id])[:, None]
    target_wall = target[:, wall_id]
    boundary = (
        F.max_pool2d(target_wall[:, None], 3, stride=1, padding=1)[:, 0]
        - -F.max_pool2d(-target_wall[:, None], 3, stride=1, padding=1)[:, 0]
    ).clamp(0.0, 1.0)
    if not boundary.any():
        return predicted.sum() * 0.0
    return (
        (1.0 - wall_probability[:, 0] * boundary) * boundary * footprint[:, 0]
    ).sum() / boundary.sum().clamp_min(1.0)


def _opening_wall_loss(
    predicted: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    opening_channels = torch.tensor(
        [class_id - 1 for class_id in taxonomy.opening_ids],
        device=predicted.device,
        dtype=torch.long,
    )
    wall_channel = taxonomy.class_id("wall") - 1
    if opening_channels.numel() == 0:
        return predicted.sum() * 0.0
    opening = torch.sigmoid(predicted[:, opening_channels]).amax(dim=1, keepdim=True)
    wall = torch.sigmoid(predicted[:, wall_channel])[:, None]
    near_wall = F.max_pool2d(wall, 5, stride=1, padding=2)
    unsupported = opening * (1.0 - near_wall) * footprint
    return (unsupported + 0.1 * opening * (1.0 - footprint)).sum() / footprint.sum().clamp_min(1.0)


def cfm_structural_loss(
    logits_x1: torch.Tensor,
    target_x1: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    weights: CFMStructuralWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match the implied endpoint to semantic structure at every flow time."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if logits_x1.shape != target_x1.shape:
        raise ValueError("predicted and target endpoint shapes must match")
    if footprint.shape != logits_x1[:, :1].shape:
        raise ValueError("footprint must match endpoint spatial shape")
    weights = weights or CFMStructuralWeights()
    groups = control_channel_groups(taxonomy)
    family_components: dict[str, torch.Tensor] = {}
    total = logits_x1.sum() * 0.0
    for name in ("room", "wall", "opening", "furniture"):
        channels = groups[name].to(logits_x1.device)
        bce, dice = _family_bce_dice(
            logits_x1,
            target_x1,
            footprint,
            channels,
        )
        family_components[f"{name}_bce"] = bce.detach()
        family_components[f"{name}_dice"] = dice.detach()
        total = (
            total
            + weights.family_bce * bce
            + weights.family_dice * dice
        )
    opening_wall = _opening_wall_loss(logits_x1, footprint, taxonomy)
    wall_boundary = _wall_boundary_loss(
        logits_x1,
        target_x1,
        footprint,
        taxonomy,
    )
    total = (
        total
        + weights.opening_wall * opening_wall
        + weights.wall_boundary * wall_boundary
    )
    components = {
        **family_components,
        "opening_wall": opening_wall.detach(),
        "wall_boundary": wall_boundary.detach(),
        "cfm_structural": total.detach(),
    }
    return total, components


__all__ = [
    "CFMStructuralWeights",
    "cfm_structural_loss",
    "predicted_x1",
]

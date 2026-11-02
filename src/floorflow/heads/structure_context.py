"""Typed geometric context for room-conditioned structure refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F

STRUCTURE_CONTEXT_NAMES = (
    "room_boundary",
    "owner_uncertainty",
    "second_owner",
    "wall_proximity",
    "footprint_boundary",
    "opening_wall_affinity",
)


def _spatial_disagreement(probability: torch.Tensor) -> torch.Tensor:
    horizontal = 1.0 - (
        probability[:, :, :, 1:] * probability[:, :, :, :-1]
    ).sum(dim=1)
    vertical = 1.0 - (
        probability[:, :, 1:, :] * probability[:, :, :-1, :]
    ).sum(dim=1)
    horizontal = F.pad(
        horizontal[:, None],
        (0, 1, 0, 0),
        mode="replicate",
    )[:, 0]
    vertical = F.pad(
        vertical[:, None],
        (0, 0, 0, 1),
        mode="replicate",
    )[:, 0]
    return torch.maximum(horizontal, vertical).clamp(0.0, 1.0)


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4:
        mask = mask[:, 0]
    mask = mask.float()
    eroded = -F.max_pool2d(
        -mask[:, None],
        3,
        stride=1,
        padding=1,
    )
    return (mask[:, None] - eroded).clamp(0.0, 1.0)


def build_structure_context(
    room_probability: torch.Tensor,
    structure_control: torch.Tensor,
    room_count: int,
    opening_count: int,
    *,
    proximity_radius: int = 2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build named boundary, ownership, and opening-host channels."""

    if room_probability.ndim != 4:
        raise ValueError("room probability must be [B,C,H,W]")
    if structure_control.shape[-2:] != room_probability.shape[-2:]:
        raise ValueError("room and structure controls must share a resolution")
    if room_probability.shape[1] != int(room_count):
        raise ValueError("room probability width does not match room count")
    probability = room_probability.float().clamp_min(1.0e-8)
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(
        1.0e-8
    )
    top_two = probability.topk(2, dim=1).values
    room_boundary = _spatial_disagreement(probability)
    owner_uncertainty = 1.0 - top_two[:, 0]
    second_owner = top_two[:, 1]
    wall = structure_control[
        :,
        int(room_count) : int(room_count) + 1,
    ].float()
    opening = structure_control[
        :,
        int(room_count) + 1 : int(room_count) + 2,
    ].float()
    footprint = structure_control[
        :,
        int(room_count) + 2 + int(opening_count) :
        int(room_count) + 3 + int(opening_count),
    ].float()
    wall_proximity = F.max_pool2d(
        wall,
        kernel_size=2 * int(proximity_radius) + 1,
        stride=1,
        padding=int(proximity_radius),
    ).clamp(0.0, 1.0)
    footprint_boundary = _boundary(footprint)
    opening_wall_affinity = opening * wall_proximity
    context = torch.cat(
        [
            room_boundary[:, None],
            owner_uncertainty[:, None],
            second_owner[:, None],
            wall_proximity,
            footprint_boundary,
            opening_wall_affinity,
        ],
        dim=1,
    )
    return context, {
        "room_boundary": room_boundary.mean(),
        "owner_uncertainty": owner_uncertainty.mean(),
        "wall_proximity": wall_proximity.mean(),
        "opening_wall_affinity": opening_wall_affinity.mean(),
    }


__all__ = ["STRUCTURE_CONTEXT_NAMES", "build_structure_context"]

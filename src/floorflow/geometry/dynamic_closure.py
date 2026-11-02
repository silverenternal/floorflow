"""Confidence-guided wall closure derived from predicted geometry."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.heads.structured_objects import partition_wall_mask


def closure_radius_from_geometry(
    wall: torch.Tensor,
    room_labels: torch.Tensor,
    footprint: torch.Tensor,
    *,
    max_radius: int | None = None,
) -> torch.Tensor:
    """Estimate a per-case closure radius from unsupported room seams."""

    if wall.ndim == 3:
        wall = wall.bool()
    else:
        wall = wall[:, 0].bool()
    if room_labels.ndim == 4:
        room_labels = room_labels[:, 0]
    if footprint.ndim == 3:
        footprint = footprint.bool()
    else:
        footprint = footprint[:, 0].bool()
    if wall.shape != room_labels.shape or wall.shape != footprint.shape:
        raise ValueError("wall, room_labels, and footprint must share [B,H,W]")
    scale_radius = max(
        1,
        round(min(footprint.shape[-2:]) * 0.012),
    )
    limit = int(max_radius) if max_radius is not None else scale_radius
    if limit <= 0:
        raise ValueError("max_radius must be positive")
    seam = partition_wall_mask(room_labels, radius=1) & footprint
    gap = seam & ~wall
    gap_fraction = (
        gap.flatten(1).float().sum(dim=1)
        / seam.flatten(1).float().sum(dim=1).clamp_min(1.0)
    )
    wall_fragment = (
        F.max_pool2d(wall.float()[:, None], 3, 1, 1)[:, 0]
        - -F.max_pool2d(-wall.float()[:, None], 3, 1, 1)[:, 0]
    ).clamp(0.0, 1.0)
    fragment_fraction = (
        (wall_fragment * footprint).flatten(1).sum(dim=1)
        / wall.flatten(1).float().sum(dim=1).clamp_min(1.0)
    )
    strength = (gap_fraction + 0.25 * fragment_fraction).clamp(0.0, 1.0)
    return torch.ceil(strength * limit).long().clamp(0, limit)


def apply_dynamic_wall_closure(
    wall: torch.Tensor,
    room_labels: torch.Tensor,
    footprint: torch.Tensor,
    *,
    max_radius: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply scale-adaptive morphological closing only when seams are unsupported."""

    if wall.ndim == 4:
        wall = wall[:, 0]
    if room_labels.ndim == 4:
        room_labels = room_labels[:, 0]
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    radii = closure_radius_from_geometry(
        wall,
        room_labels,
        footprint,
        max_radius=max_radius,
    )
    closed = wall.bool().clone()
    for radius in radii.unique().tolist():
        if int(radius) <= 0:
            continue
        selected = radii == int(radius)
        value = wall[selected].float()[:, None]
        value = F.max_pool2d(
            value,
            2 * int(radius) + 1,
            stride=1,
            padding=int(radius),
        )
        value = -F.max_pool2d(
            -value,
            2 * int(radius) + 1,
            stride=1,
            padding=int(radius),
        )
        closed[selected] = value[:, 0] > 0.5
    closed = closed & footprint.bool()
    return closed.long(), radii


__all__ = [
    "apply_dynamic_wall_closure",
    "closure_radius_from_geometry",
]

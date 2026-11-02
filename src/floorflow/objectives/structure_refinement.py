"""Typed relation losses for the room-to-structure bridge."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.geometry_heads import (
    focal_heatmap_loss,
    wall_junction_targets,
)
from floorflow.objectives.structural_consistency import (
    opening_wall_affinity_loss,
    room_wall_edge_loss,
)


def room_boundary_target(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    room_ids: tuple[int, ...],
) -> torch.Tensor:
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    rooms = torch.isin(
        labels,
        torch.tensor(room_ids, device=labels.device),
    ) & footprint.bool()
    mask = rooms.float()[:, None]
    eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (mask - eroded).clamp(0.0, 1.0)


def predicted_opening_wall_containment(
    predictions: dict[str, torch.Tensor],
    footprint: torch.Tensor,
    *,
    radius: int = 2,
) -> torch.Tensor:
    opening = predictions["opening_logits"].sigmoid()
    wall = predictions["wall_logits"].sigmoid()
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    target_size = footprint.shape[-2:]
    if opening.shape[-2:] != target_size:
        opening = F.interpolate(
            opening,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    if wall.shape[-2:] != target_size:
        wall = F.interpolate(
            wall,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    near_wall = F.max_pool2d(
        wall,
        kernel_size=2 * int(radius) + 1,
        stride=1,
        padding=int(radius),
    )
    area = footprint.float()
    loss = opening * (1.0 - near_wall) * area[:, None]
    return loss.sum() / area.sum().clamp_min(1.0)


def typed_structure_relation_loss(
    predictions: dict[str, torch.Tensor],
    room_logits: torch.Tensor,
    structure_target: torch.Tensor,
    target_labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    room_count: int,
    opening_host_weight: float = 0.25,
    room_edge_weight: float = 0.25,
    junction_weight: float = 0.5,
    boundary_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise typed geometry relations without replacing pixel losses."""

    target_wall = structure_target[
        :,
        int(room_count) : int(room_count) + 1,
    ]
    opening_for_affinity = F.interpolate(
        predictions["opening_logits"],
        size=target_labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    opening_affinity = opening_wall_affinity_loss(
        opening_for_affinity,
        torch.zeros_like(opening_for_affinity[:, 0]),
        target_labels,
        footprint,
        taxonomy,
    )
    opening_host = predicted_opening_wall_containment(
        predictions,
        footprint,
    )
    room_edges = room_wall_edge_loss(
        F.interpolate(
            room_logits,
            size=target_wall.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ),
        F.interpolate(
            target_labels[:, None].float(),
            size=target_wall.shape[-2:],
            mode="nearest",
        )[:, 0].long(),
        F.interpolate(
            footprint[:, None].float(),
            size=target_wall.shape[-2:],
            mode="nearest",
        )[:, 0],
        taxonomy,
    )
    zero = predictions["wall_logits"].sum() * 0.0
    junction = zero
    boundary = zero
    if "junction_logits" in predictions:
        junction_footprint = F.interpolate(
            footprint[:, None].float()
            if footprint.ndim == 3
            else footprint.float(),
            size=target_wall.shape[-2:],
            mode="nearest",
        )[:, 0]
        junction_target, degree_target = wall_junction_targets(
            target_wall,
            junction_footprint,
        )
        junction = focal_heatmap_loss(
            predictions["junction_logits"],
            junction_target,
        )
        if "junction_degree_logits" in predictions:
            valid = degree_target > 0
            if valid.any():
                logits = predictions["junction_degree_logits"].permute(
                    0,
                    2,
                    3,
                    1,
                )
                junction = junction + F.cross_entropy(
                    logits[valid],
                    degree_target[valid],
                )
    if "room_boundary_logits" in predictions:
        boundary_labels = F.interpolate(
            target_labels[:, None].float(),
            size=target_wall.shape[-2:],
            mode="nearest",
        )[:, 0].long()
        boundary_footprint = F.interpolate(
            footprint[:, None].float(),
            size=target_wall.shape[-2:],
            mode="nearest",
        )[:, 0]
        boundary = focal_heatmap_loss(
            predictions["room_boundary_logits"],
            room_boundary_target(
                boundary_labels,
                boundary_footprint,
                taxonomy.room_ids,
            ),
        )
    total = (
        float(opening_host_weight) * opening_host
        + float(room_edge_weight) * room_edges
        + float(junction_weight) * junction
        + float(boundary_weight) * boundary
        + 0.25 * opening_affinity
    )
    return total, {
        "structure_opening_host_loss": opening_host.detach(),
        "structure_opening_affinity_loss": opening_affinity.detach(),
        "structure_room_edge_loss": room_edges.detach(),
        "structure_junction_loss": junction.detach(),
        "structure_room_boundary_loss": boundary.detach(),
        "structure_relation_loss": total.detach(),
    }


canonical_plan_proxy_loss = typed_structure_relation_loss


__all__ = [
    "canonical_plan_proxy_loss",
    "predicted_opening_wall_containment",
    "room_boundary_target",
    "typed_structure_relation_loss",
]

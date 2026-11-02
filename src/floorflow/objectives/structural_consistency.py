"""Differentiable wall, opening, room-edge, and furniture consistency terms."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy


def _foreground_probability(
    logits: torch.Tensor,
    background_logits: torch.Tensor,
) -> torch.Tensor:
    scores = torch.cat([background_logits[:, None], logits], dim=1)
    return 1.0 - scores.softmax(dim=1)[:, 0]


def opening_wall_affinity_loss(
    opening_logits: torch.Tensor,
    opening_background_logits: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    radius: int = 2,
) -> torch.Tensor:
    wall = (target == taxonomy.class_id("wall")).float()[:, None]
    kernel = 2 * int(radius) + 1
    near_wall = F.max_pool2d(wall, kernel, stride=1, padding=radius) > 0
    opening_probability = _foreground_probability(
        opening_logits,
        opening_background_logits,
    )
    far_opening = opening_probability * (~near_wall[:, 0]).float()
    return (far_opening * footprint).sum() / footprint.sum().clamp_min(1.0)


def room_wall_edge_loss(
    room_logits: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    probability = room_logits.softmax(dim=1)
    disagreement = 1.0 - (
        probability[:, :, :, 1:] * probability[:, :, :, :-1]
    ).sum(dim=1)
    wall = target == taxonomy.class_id("wall")
    wall_edges = (
        (wall[:, :, 1:] | wall[:, :, :-1])
        & footprint[:, :, 1:].bool()
        & footprint[:, :, :-1].bool()
    )
    if not wall_edges.any():
        return room_logits.sum() * 0.0
    probability = disagreement.float().clamp(1.0e-4, 1.0 - 1.0e-4)
    expected = wall_edges.float()
    elementwise = -(
        expected * probability.log()
        + (1.0 - expected) * (1.0 - probability).log()
    )
    return elementwise.masked_select(wall_edges).mean()


def furniture_containment_loss(
    furniture_logits: torch.Tensor,
    furniture_background_logits: torch.Tensor,
    room_logits: torch.Tensor,
    footprint: torch.Tensor,
    room_probability: torch.Tensor | None = None,
) -> torch.Tensor:
    furniture_probability = _foreground_probability(
        furniture_logits,
        furniture_background_logits,
    )
    if room_probability is None:
        room_probability = room_logits.softmax(dim=1).sum(dim=1)
    else:
        room_probability = room_probability.to(furniture_probability.dtype)
    outside_room = (1.0 - room_probability).clamp(0.0, 1.0)
    return (furniture_probability * outside_room * footprint).sum() / (
        footprint.sum().clamp_min(1.0)
    )


def structural_consistency_loss(
    room_logits: torch.Tensor,
    objects: dict[str, torch.Tensor],
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    room_probability: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    opening = opening_wall_affinity_loss(
        objects["opening_logits"],
        objects.get(
            "opening_background_logits",
            torch.zeros_like(objects["opening_logits"][:, 0]),
        ),
        target,
        footprint,
        taxonomy,
    )
    room_edges = room_wall_edge_loss(
        room_logits,
        target,
        footprint,
        taxonomy,
    )
    furniture = furniture_containment_loss(
        objects["furniture_logits"],
        objects.get(
            "furniture_background_logits",
            torch.zeros_like(objects["furniture_logits"][:, 0]),
        ),
        room_logits,
        footprint,
        room_probability=room_probability,
    )
    total = opening + 0.25 * room_edges + 0.25 * furniture
    return total, {
        "opening_wall_affinity": opening.detach(),
        "room_wall_edge": room_edges.detach(),
        "furniture_room_containment": furniture.detach(),
        "structural_consistency": total.detach(),
    }

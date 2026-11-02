"""Context-gated legalization of room, opening, furniture and wall layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.geometry.dynamic_closure import apply_dynamic_wall_closure
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.representation.layered_floorplan import LayeredFloorplan


def legalize_layers(
    layers: LayeredFloorplan,
    footprint: torch.Tensor,
    taxonomy,
    *,
    opening_wall_radius: int = 2,
    furniture_room_radius: int = 0,
    closure_radius: int = 0,
    dynamic_closure: bool = False,
) -> LayeredFloorplan:
    """Suppress impossible objects and optionally close low-level room seams."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    footprint = footprint.bool()
    rooms = layers.room_labels
    room_ids = torch.tensor(taxonomy.room_ids, device=rooms.device)
    room_support = torch.isin(rooms, room_ids)
    wall = layers.wall.bool()
    opening = torch.isin(
        layers.opening,
        torch.tensor(taxonomy.opening_ids, device=rooms.device),
    )
    opening = torch.where(
        opening & footprint[:, 0],
        layers.opening,
        torch.zeros_like(layers.opening),
    )
    furniture_radius = max(0, int(furniture_room_radius))
    if furniture_radius:
        room_near = F.max_pool2d(
            room_support.float()[:, None],
            kernel_size=2 * furniture_radius + 1,
            stride=1,
            padding=furniture_radius,
        )[:, 0] > 0.5
    else:
        room_near = room_support
    furniture = torch.isin(
        layers.furniture,
        torch.tensor(taxonomy.furniture_ids, device=rooms.device),
    )
    furniture = torch.where(
        furniture & room_near & footprint[:, 0],
        layers.furniture,
        torch.zeros_like(layers.furniture),
    )
    if dynamic_closure:
        wall, _radii = apply_dynamic_wall_closure(
            wall,
            rooms,
            footprint,
            max_radius=closure_radius or None,
        )
        wall = wall.bool()
    elif closure_radius > 0:
        closure = partition_wall_mask(
            rooms,
            radius=int(closure_radius),
        )
        wall = (wall | closure) & footprint[:, 0]
    return LayeredFloorplan(
        room_labels=rooms,
        wall=wall.long(),
        opening=opening,
        furniture=furniture,
        footprint=footprint[:, 0],
        taxonomy=taxonomy,
    )


__all__ = ["legalize_layers"]

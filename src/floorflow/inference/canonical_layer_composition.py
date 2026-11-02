"""Shared composition from CAD layers to canonical visible labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.core import compose_known_exterior
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.representation.layered_floorplan import LayeredFloorplan


def canonical_union_labels(
    layers: LayeredFloorplan,
    taxonomy: FloorplanTaxonomy,
    *,
    radius: int = 0,
    boundary_width: int = 7,
) -> torch.Tensor:
    """Apply the shared seam, wall-band opening, and exterior contract."""

    rooms = layers.room_labels * layers.footprint.long()
    seam = partition_wall_mask(rooms, radius=int(radius))
    wall = layers.wall.bool() | (seam & layers.footprint.bool())
    wall_support = F.max_pool2d(
        wall.float()[:, None],
        kernel_size=5,
        stride=1,
        padding=2,
    )[:, 0] > 0.5
    opening = torch.where(
        wall_support,
        layers.opening,
        torch.zeros_like(layers.opening),
    )
    composed = LayeredFloorplan(
        room_labels=layers.room_labels,
        wall=wall.long(),
        opening=opening,
        furniture=layers.furniture,
        footprint=layers.footprint,
        taxonomy=taxonomy,
    )
    return compose_known_exterior(
        composed.visible_labels(),
        composed.footprint,
        boundary_width=int(boundary_width),
        opening_ids=taxonomy.opening_ids,
    )


__all__ = ["canonical_union_labels"]

"""Fuse deleted room support back into predicted CAD structure layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.layered_floorplan import LayeredFloorplan


def apply_room_support_to_visible_plan(
    visible_labels: torch.Tensor,
    keep_mask: torch.Tensor,
    room_ids: torch.Tensor,
) -> torch.Tensor:
    """Remove room support without recomputing canonical wall or opening layers."""

    if visible_labels.shape != keep_mask.shape:
        raise ValueError("visible labels and keep mask must share a shape")
    room_mask = torch.isin(visible_labels, room_ids.to(visible_labels.device))
    return torch.where(
        room_mask,
        visible_labels * keep_mask.to(visible_labels.device).long(),
        visible_labels,
    )


def fuse_support_deletion_with_structure(
    base_layers: LayeredFloorplan,
    gated_room_labels: torch.Tensor,
    structure: dict[str, torch.Tensor],
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    wall_threshold: float | None,
    opening_threshold: float | None,
) -> LayeredFloorplan:
    """Assign deleted room pixels to structure evidence instead of bare void."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    footprint = footprint.bool()
    room_support = base_layers.room_labels > 0
    deleted = room_support & (gated_room_labels <= 0) & footprint[:, 0]
    wall = base_layers.wall.bool()
    opening = base_layers.opening.long()
    if wall_threshold is not None:
        wall_probability = F.interpolate(
            structure["wall_logits"].sigmoid(),
            size=gated_room_labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        wall = wall | (
            deleted
            & (wall_probability >= float(wall_threshold))
            & footprint[:, 0]
        )
    if opening_threshold is not None:
        opening_probability = F.interpolate(
            structure["opening_logits"].sigmoid(),
            size=gated_room_labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        opening_type = F.interpolate(
            structure["opening_type_logits"],
            size=gated_room_labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        opening_ids = torch.tensor(
            taxonomy.opening_ids,
            device=gated_room_labels.device,
        )
        opening_mask = (
            deleted
            & (opening_probability >= float(opening_threshold))
            & footprint[:, 0]
        )
        opening = torch.where(
            opening_mask,
            opening_ids[opening_type],
            opening,
        )
    return LayeredFloorplan(
        room_labels=gated_room_labels,
        wall=wall.long(),
        opening=opening,
        furniture=base_layers.furniture,
        footprint=base_layers.footprint,
        taxonomy=taxonomy,
    )


__all__ = [
    "apply_room_support_to_visible_plan",
    "fuse_support_deletion_with_structure",
]

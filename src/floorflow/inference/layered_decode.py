"""Decode a refined room layer plus constrained structural overlays."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.structured_objects import constrain_structured_objects
from floorflow.inference.structured_decode import decode_structured_floorplan
from floorflow.representation.layered_floorplan import LayeredFloorplan


@torch.no_grad()
def decode_layered_floorplan(
    model,
    reference: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    room_labels: torch.Tensor | None = None,
    floor_ids: torch.Tensor | None = None,
    opening_threshold: float | torch.Tensor = 0.5,
    furniture_threshold: float | torch.Tensor = 0.5,
    wall_threshold: float = 0.5,
    object_score_scale: str = "probability",
    room_source: str = "partition",
    wall_radius: int = 0,
    support_radius: int = 2,
) -> LayeredFloorplan:
    """Keep structural heads out of the room branch and project them locally."""

    _labels, decoded_rooms, _raw_rooms, raw_objects = decode_structured_floorplan(
        model,
        reference,
        footprint,
        taxonomy,
        floor_ids=floor_ids,
        opening_threshold=opening_threshold,
        furniture_threshold=furniture_threshold,
        wall_threshold=wall_threshold,
        object_score_scale=object_score_scale,
        structural_gate=False,
        room_source=room_source,
        return_debug=True,
    )
    rooms = decoded_rooms if room_labels is None else room_labels
    objects = constrain_structured_objects(
        raw_objects,
        rooms,
        footprint,
        taxonomy,
        wall_radius=wall_radius,
        support_radius=support_radius,
        use_predicted_walls=True,
        require_compatible_furniture=True,
    )
    outside = F.pad(
        (~footprint.bool()).float()[:, None],
        (1, 1, 1, 1),
        mode="constant",
        value=1.0,
    )
    exterior = footprint.bool() & (
        F.max_pool2d(outside, kernel_size=3, stride=1)[:, 0] > 0.5
    )
    return LayeredFloorplan(
        room_labels=rooms * footprint.long(),
        wall=(objects["wall"].bool() | exterior).long() * footprint.long(),
        opening=objects["opening"].long() * footprint.long(),
        furniture=objects["furniture"].long() * footprint.long(),
        footprint=footprint,
        taxonomy=taxonomy,
    )


__all__ = ["decode_layered_floorplan"]

"""Three facades over the unified typed floorplan state."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from floorflow.representation.typed_floorplan_state import TypedFloorplanState


@dataclass
class RoomInstanceLayer:
    ids: torch.Tensor | None
    features: torch.Tensor | None
    existence: torch.Tensor | None
    classes: torch.Tensor | None


@dataclass
class TopologyLayer:
    adjacency: torch.Tensor | None
    wall_exists: torch.Tensor | None
    wall_types: torch.Tensor | None
    opening_exists: torch.Tensor | None
    opening_hosts: torch.Tensor | None
    junction_coordinates: torch.Tensor | None
    junction_degrees: torch.Tensor | None


@dataclass
class GeometryLayer:
    room_polygons: torch.Tensor | None
    wall_endpoints: torch.Tensor | None
    opening_centers: torch.Tensor | None
    opening_widths: torch.Tensor | None


def split_stage_layers(
    state: TypedFloorplanState,
) -> tuple[RoomInstanceLayer, TopologyLayer, GeometryLayer]:
    return (
        RoomInstanceLayer(
            state.room_instance_ids,
            state.room_features,
            state.room_exists,
            state.room_class,
        ),
        TopologyLayer(
            state.room_adjacency_logits,
            state.wall_exists,
            state.wall_exterior_logits,
            state.opening_exists,
            state.opening_host_logits,
            state.junction_coordinates,
            state.junction_degree_logits,
        ),
        GeometryLayer(
            state.room_polygons,
            state.wall_endpoints,
            state.opening_centers,
            state.opening_widths,
        ),
    )


def stage_layers_dict(state: TypedFloorplanState) -> dict[str, dict]:
    rooms, topology, geometry = split_stage_layers(state)
    return {
        "room_instance": {
            field.name: getattr(rooms, field.name)
            for field in fields(rooms)
        },
        "topology": {
            field.name: getattr(topology, field.name)
            for field in fields(topology)
        },
        "geometry": {
            field.name: getattr(geometry, field.name)
            for field in fields(geometry)
        },
    }


__all__ = [
    "GeometryLayer",
    "RoomInstanceLayer",
    "TopologyLayer",
    "split_stage_layers",
    "stage_layers_dict",
]

from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.wall_centerline_fusion import fuse_wall_centerline


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=("background", "wall", "room:LivingRoom", "room:Bedroom"),
        room_map={"LivingRoom": "room:LivingRoom", "Bedroom": "room:Bedroom"},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_add_mode_bridges_rooms_with_centerline_wall() -> None:
    taxonomy = _taxonomy()
    living = taxonomy.class_id("room:LivingRoom")
    bedroom = taxonomy.class_id("room:Bedroom")
    wall = taxonomy.class_id("wall")
    labels = torch.full((1, 16, 16), living)
    labels[:, :, 8:] = bedroom
    labels[:, :, 8] = wall
    rooms = labels.clone()
    rooms[:, :, 8] = living
    footprint = torch.ones_like(labels, dtype=torch.bool)
    probability = torch.zeros(1, 16, 16)
    probability[:, :, 8] = 1.0
    fused, metrics = fuse_wall_centerline(
        labels,
        rooms,
        footprint,
        probability,
        taxonomy,
        mode="add",
    )
    assert (fused == wall).any()
    assert metrics["wall_pixels_added"] >= 0.0


def test_consensus_removes_off_centerline_wall_pixels() -> None:
    taxonomy = _taxonomy()
    living = taxonomy.class_id("room:LivingRoom")
    wall = taxonomy.class_id("wall")
    labels = torch.full((1, 16, 16), living)
    labels[:, :, 7:10] = wall
    rooms = torch.full_like(labels, living)
    footprint = torch.ones_like(labels, dtype=torch.bool)
    probability = torch.zeros(1, 16, 16)
    probability[:, :, 8] = 1.0
    fused, metrics = fuse_wall_centerline(
        labels,
        rooms,
        footprint,
        probability,
        taxonomy,
        dilation=0,
        mode="consensus",
    )
    assert metrics["wall_pixels_removed"] > 0
    assert (fused == wall).sum() < (labels == wall).sum()

from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.opening_evidence_fusion import fuse_opening_evidence


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "door:Swing Beside",
            "room:LivingRoom",
        ),
        room_map={"LivingRoom": "room:LivingRoom"},
        furniture_map={},
        door_map={"Swing Beside": "door:Swing Beside"},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_opening_evidence_only_replaces_wall_band() -> None:
    taxonomy = _taxonomy()
    wall = taxonomy.class_id("wall")
    room = taxonomy.class_id("room:LivingRoom")
    door = taxonomy.class_id("door:Swing Beside")
    labels = torch.full((1, 16, 16), room)
    labels[:, :, 8] = wall
    footprint = torch.ones_like(labels, dtype=torch.bool)
    opening = torch.zeros_like(labels)
    opening[0, 7:10, 8] = door
    opening[0, 1, 1] = door
    fused = fuse_opening_evidence(labels, footprint, opening, taxonomy)
    assert (fused[0, 8, 8] == door).all()
    assert fused[0, 1, 1] == room

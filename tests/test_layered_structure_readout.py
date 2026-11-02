from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.layered_structure_readout import (
    build_layered_structure_plan,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=("background", "wall", "room:LivingRoom", "room:Bedroom"),
        room_map={"LivingRoom": "room:LivingRoom", "Bedroom": "room:Bedroom"},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_layered_plan_keeps_room_polygons_with_independent_wall_map() -> None:
    taxonomy = _taxonomy()
    living = taxonomy.class_id("room:LivingRoom")
    bedroom = taxonomy.class_id("room:Bedroom")
    wall = taxonomy.class_id("wall")
    rooms = np.zeros((32, 32), dtype=np.int64)
    rooms[2:30, 2:15] = living
    rooms[2:30, 15:30] = bedroom
    footprint = np.ones_like(rooms, dtype=bool)
    walls = np.zeros_like(rooms)
    walls[:, 15] = wall
    plan = build_layered_structure_plan(rooms, walls, footprint, taxonomy)
    assert plan.rooms
    assert plan.room_source == "layered_room_partition"
    assert plan.walls

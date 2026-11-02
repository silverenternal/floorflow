"""Tests for edit-local topology projection."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.vector_edit_protocol import VectorPlan, VectorRoom
from floorflow.inference.vector_topology_projection import (
    project_edit_topology,
    repair_edited_room_polygons,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "door:Swing Beside",
            "room:LivingRoom",
            "room:Bedroom",
        ),
        room_map={"LivingRoom": "room:LivingRoom", "Bedroom": "room:Bedroom"},
        furniture_map={},
        door_map={"Swing Beside": "door:Swing Beside"},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_projection_preserves_outside_and_restores_wall_seam() -> None:
    taxonomy = _taxonomy()
    before = np.ones((32, 32), dtype=np.int64)
    before[2:-2, 2:-2] = 3
    before[2:-2, 16:] = 4
    before[:, 15:17] = 1
    after = before.copy()
    after[8:24, 15:17] = 3
    room_labels = np.zeros_like(before)
    room_labels[3:-3, 3:15] = 3
    room_labels[3:-3, 17:-3] = 4
    room_labels[15:18, 14:18] = 2
    footprint = np.ones_like(before, dtype=bool)
    region = np.zeros_like(before, dtype=bool)
    region[8:24, 12:20] = True

    result = project_edit_topology(
        before,
        after,
        room_labels,
        region,
        footprint,
        taxonomy,
    )

    assert result.outside_changed_pixels == 0
    assert result.projected_wall_pixels > 0
    assert np.any(result.labels[8:24, 15:17] == 1)


def test_projection_can_disable_wall_repair() -> None:
    taxonomy = _taxonomy()
    labels = np.ones((16, 16), dtype=np.int64)
    labels[2:-2, 2:-2] = 3
    room_labels = labels.copy()
    region = np.zeros_like(labels, dtype=bool)
    region[4:12, 4:12] = True
    result = project_edit_topology(
        labels,
        labels,
        room_labels,
        region,
        np.ones_like(labels, dtype=bool),
        taxonomy,
        fill_holes=False,
        project_walls=False,
    )

    assert result.changed_pixels == 0
    assert result.projected_wall_pixels == 0


def test_vector_polygon_repair_preserves_unedited_rooms() -> None:
    edited = VectorRoom(
        label="room:LivingRoom",
        polygon=Polygon([(2, 2), (12, 2), (2, 12)]),
        mask=np.ones((16, 16), dtype=bool),
    )
    untouched_polygon = Polygon([(10, 10), (14, 10), (14, 14), (10, 14)])
    untouched = VectorRoom(
        label="room:Bedroom",
        polygon=untouched_polygon,
        mask=np.zeros((16, 16), dtype=bool),
    )
    plan = VectorPlan(
        rooms=(edited, untouched),
        walls=(),
        openings=(),
        wall_mask=np.zeros((16, 16), dtype=bool),
        opening_mask=np.zeros((16, 16), dtype=bool),
        source=None,
    )
    region = np.zeros((16, 16), dtype=bool)
    region[1:9, 1:9] = True

    result = repair_edited_room_polygons(
        plan,
        region,
        np.ones((16, 16), dtype=bool),
    )

    assert result["repaired_rooms"] == 1
    assert result["plan"].rooms[1].polygon.equals_exact(untouched_polygon, 1.0e-9)

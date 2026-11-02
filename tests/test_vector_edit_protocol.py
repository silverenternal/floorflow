"""Tests for FEVP-2 vector edit preservation."""

from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.vector_edit_protocol import (
    extract_vector_plan,
    vector_preservation_metrics,
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


def _layout() -> tuple[np.ndarray, np.ndarray]:
    labels = np.ones((64, 64), dtype=np.uint8)
    labels[2:-2, 2:-2] = 0
    labels[:, 31:33] = 1
    labels[2:7, 15:18] = 2
    room_labels = np.zeros_like(labels)
    room_labels[3:-3, 3:31] = 3
    room_labels[3:-3, 33:-3] = 4
    return labels, room_labels


def test_vector_plan_is_parsable() -> None:
    labels, room_labels = _layout()
    plan = extract_vector_plan(labels, room_labels, _taxonomy())

    assert plan.rooms
    assert plan.walls
    assert plan.openings


def test_identical_vector_plans_preserve_all_unaffected_elements() -> None:
    labels, room_labels = _layout()
    edit_region = np.zeros(labels.shape, dtype=bool)
    edit_region[20:40, 35:55] = True
    metrics = vector_preservation_metrics(
        labels,
        room_labels,
        labels,
        room_labels,
        edit_region,
        _taxonomy(),
    )

    assert metrics["outside_room_level_exact"] == 1.0
    assert metrics["outside_wall_exact"] == 1.0
    assert metrics["outside_opening_exact"] == 1.0
    assert metrics["outside_edge_exact"] == 1.0
    assert metrics["locality_violation_area"] == 0.0


def test_room_label_change_outside_region_is_detected() -> None:
    labels, room_labels = _layout()
    after_room_labels = room_labels.copy()
    after_room_labels[room_labels == 3] = 4
    edit_region = np.zeros(labels.shape, dtype=bool)
    edit_region[20:40, 35:55] = True
    metrics = vector_preservation_metrics(
        labels,
        room_labels,
        labels,
        after_room_labels,
        edit_region,
        _taxonomy(),
    )

    assert metrics["outside_room_label_exact"] < 1.0
    assert metrics["outside_room_level_exact"] < 1.0

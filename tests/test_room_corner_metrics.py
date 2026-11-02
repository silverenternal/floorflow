from __future__ import annotations

import inspect

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_corner_metrics import room_corner_angle_metrics


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=("background", "wall", "room:LivingRoom"),
        room_map={"LivingRoom": "room:LivingRoom"},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_raster2seq_corner_and_angle_tolerances_are_frozen() -> None:
    parameters = inspect.signature(room_corner_angle_metrics).parameters
    assert parameters["corner_tolerance_px"].default == 10.0
    assert parameters["angle_tolerance_degrees"].default == 5.0


def test_identical_rooms_have_perfect_corner_and_angle_scores() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((24, 24), dtype=np.int64)
    labels[2:22, 2:22] = taxonomy.class_id("wall")
    labels[3:21, 3:21] = taxonomy.class_id("room:LivingRoom")
    metrics = room_corner_angle_metrics(labels, labels, taxonomy)
    assert metrics["corner_f1"] == 1.0
    assert metrics["angle_f1"] == 1.0
    assert metrics["orthogonal_turn_rate"] == 1.0


def test_shifted_room_has_no_corner_match() -> None:
    taxonomy = _taxonomy()
    predicted = np.zeros((48, 48), dtype=np.int64)
    target = np.zeros((48, 48), dtype=np.int64)
    predicted[2:22, 2:22] = taxonomy.class_id("wall")
    predicted[3:21, 3:21] = taxonomy.class_id("room:LivingRoom")
    target[26:46, 26:46] = taxonomy.class_id("wall")
    target[27:45, 27:45] = taxonomy.class_id("room:LivingRoom")
    metrics = room_corner_angle_metrics(predicted, target, taxonomy)
    assert metrics["corner_f1"] == 0.0

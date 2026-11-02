from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.predicted_control_fidelity import (
    predicted_control_fidelity_metrics,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "room:LivingRoom",
            "room:Bedroom",
        ),
        room_map={"LivingRoom": "room:LivingRoom", "Bedroom": "room:Bedroom"},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_identity_fidelity_scores_one_under_explicit_plan() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    footprint = np.zeros_like(labels)
    footprint[4:60, 4:60] = 1
    labels[7:57, 7:31] = taxonomy.class_id("room:LivingRoom")
    labels[7:57, 33:57] = taxonomy.class_id("room:Bedroom")
    labels[4:60, 4:8] = taxonomy.class_id("wall")
    labels[4:60, 56:60] = taxonomy.class_id("wall")
    labels[4:8, 4:60] = taxonomy.class_id("wall")
    labels[56:60, 4:60] = taxonomy.class_id("wall")
    labels[4:60, 31:33] = taxonomy.class_id("wall")
    metrics = predicted_control_fidelity_metrics(
        labels,
        labels,
        footprint,
        taxonomy,
        reconcile_gap_fraction=0.04,
    )
    assert metrics["room_macro_iou"] == 1.0
    assert metrics["fpeval_wall_f1"] == 1.0
    assert metrics["node_f1"] == 1.0
    assert metrics["edge_f1"] == 1.0
    assert metrics["ged"] == 0.0


def test_room_decomposition_is_optional_and_consistent_with_identity() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((32, 32), dtype=np.int64)
    footprint = np.zeros_like(labels)
    footprint[2:30, 2:30] = 1
    labels[4:28, 4:15] = taxonomy.class_id("room:LivingRoom")
    labels[4:28, 16:28] = taxonomy.class_id("room:Bedroom")
    metrics = predicted_control_fidelity_metrics(
        labels,
        labels,
        footprint,
        taxonomy,
        reconcile_gap_fraction=0.04,
        include_room_decomposition=True,
    )
    decomposition = metrics["room_error_decomposition"]
    assert decomposition["strict_macro_iou"] == 1.0
    assert decomposition["allocation"]["correct_identity_fraction"] == 1.0

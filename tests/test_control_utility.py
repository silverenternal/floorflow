from __future__ import annotations

from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.control_utility import (
    adjacency_f1,
    intent_preservation,
    room_component_topology,
)

TAXONOMY = Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json")


def test_control_utility_reports_preservation_and_topology() -> None:
    taxonomy = FloorplanTaxonomy.read(TAXONOMY)
    room = taxonomy.room_ids[0]
    opening = taxonomy.opening_ids[0]
    control = np.zeros((16, 16), dtype=np.uint8)
    control[:8] = room
    prediction = control.copy()
    prediction[8] = room
    prediction[7, :] = opening
    preservation = intent_preservation(prediction, control, taxonomy)
    topology = room_component_topology(prediction, taxonomy)
    assert preservation["control_room_pixel_preservation"] > 0.8
    assert topology["room_components"] == 2
    assert topology["room_adjacency_edges"] == 1


def test_adjacency_f1_identical_plan_is_one() -> None:
    taxonomy = FloorplanTaxonomy.read(TAXONOMY)
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[:, :8] = taxonomy.room_ids[0]
    labels[:, 8:] = taxonomy.room_ids[1]
    labels[7:9, 8] = taxonomy.opening_ids[0]
    result = adjacency_f1(labels, labels, taxonomy)
    assert result["room_adjacency_f1"] == 1.0

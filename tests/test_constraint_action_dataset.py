"""Tests for constraint-action graph serialization."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from floorflow.data.constraint_action_dataset import (
    ActionRecord,
    control_graph_state,
    hard_feasibility,
)
from floorflow.data.taxonomy import build_taxonomy


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 3, "Kitchen": 3, "Bath": 3}),
            "furniture": Counter({"Sink": 3}),
            "door": Counter({"Swing Beside": 3}),
            "window": Counter({"Regular": 3}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _controls(taxonomy):
    rooms = torch.full((len(taxonomy.room_ids), 8, 8), -4.0)
    rooms[0, 1:4, 1:4] = 4.0
    rooms[1, 1:4, 4:7] = 4.0
    structure = torch.zeros(
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        8,
        8,
    )
    structure[len(taxonomy.room_ids), 0:8, 0] = 1.0
    furniture = torch.zeros(
        len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
        8,
        8,
    )
    support = torch.ones(len(taxonomy.room_ids) + 4, 8, 8)
    program = torch.zeros(len(taxonomy.room_ids), dtype=torch.bool)
    program[:2] = True
    return {
        "room_logits": rooms,
        "structure": structure,
        "furniture": furniture,
        "support": support,
        "program": program,
    }


def test_control_graph_state_is_normalized() -> None:
    taxonomy = _taxonomy()
    state = control_graph_state(_controls(taxonomy), taxonomy)
    assert state.node_features.shape == (len(taxonomy.room_ids), 8)
    assert state.adjacency.shape == (
        len(taxonomy.room_ids),
        len(taxonomy.room_ids),
    )
    assert torch.allclose(
        state.adjacency.sum(dim=1),
        torch.ones(len(taxonomy.room_ids)),
        atol=1.0e-5,
    )
    assert torch.isfinite(state.global_features).all()


def test_hard_feasibility_flags_local_change() -> None:
    taxonomy = _taxonomy()
    before = _controls(taxonomy)
    after = {key: value.clone() for key, value in before.items()}
    after["room_logits"][1, 1:3, 1:3] = 4.0
    record = ActionRecord(
        name="case",
        split="train",
        source="memory",
        operation="room_type",
        seed=1,
        bbox=(1, 1, 2, 2),
        control_size=(8, 8),
        changed_pixels=4,
    )
    flags = hard_feasibility(before, after, record, taxonomy)
    assert flags["arrays_finite"]
    assert flags["program_nonempty"]
    assert flags["room_support_nonempty"]
    assert flags["change_inside_edit_region"]
    assert flags["room_within_footprint"]
    assert flags["structure_within_footprint"]
    assert flags["openings_near_wall"]
    assert flags["furniture_within_footprint"]


def test_hard_feasibility_detects_content_outside_footprint() -> None:
    taxonomy = _taxonomy()
    before = _controls(taxonomy)
    after = {key: value.clone() for key, value in before.items()}
    after["room_logits"][1, 1:3, 1:3] = 4.0
    after["structure"][len(taxonomy.room_ids), 7, 0:8] = 1.0
    record = ActionRecord(
        name="case",
        split="train",
        source="memory",
        operation="room_type",
        seed=1,
        bbox=(1, 1, 2, 2),
        control_size=(8, 8),
        changed_pixels=4,
    )
    footprint = np.zeros((8, 8), dtype=bool)
    footprint[:7, :7] = True
    flags = hard_feasibility(
        before,
        after,
        record,
        taxonomy,
        footprint=footprint,
    )
    assert not flags["structure_within_footprint"]

"""Tests for deterministic constraint candidate materialization."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from floorflow.data.constraint_action_dataset import ActionRecord
from floorflow.data.constraint_candidates import (
    VIOLATION_FAMILIES,
    build_candidate_specs,
    materialize_candidate,
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
    rooms = torch.full((len(taxonomy.room_ids), 16, 16), -4.0)
    rooms[0, 2:7, 2:7] = 4.0
    rooms[1, 2:7, 7:12] = 4.0
    structure = torch.zeros(
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        16,
        16,
    )
    structure[len(taxonomy.room_ids), :, 0] = 1.0
    structure[len(taxonomy.room_ids), :, 7] = 1.0
    furniture = torch.zeros(
        len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
        16,
        16,
    )
    support = torch.ones(len(taxonomy.room_ids) + 4, 16, 16)
    program = torch.zeros(len(taxonomy.room_ids), dtype=torch.bool)
    program[:2] = True
    return {
        "room_logits": rooms,
        "structure": structure,
        "furniture": furniture,
        "support": support,
        "program": program,
    }


def _record() -> ActionRecord:
    return ActionRecord(
        name="case",
        split="train",
        source="memory",
        operation="room_type",
        seed=19,
        bbox=(2, 2, 5, 5),
        control_size=(16, 16),
        changed_pixels=25,
    )


def test_candidate_specs_cover_legal_and_seven_violation_families() -> None:
    specs = build_candidate_specs(_record())
    assert len(specs) == 9
    assert len({spec.candidate_id for spec in specs}) == 9
    assert {spec.role for spec in specs} == {
        "positive",
        "legal_alternative",
        "near_violation",
    }
    assert {
        spec.violation_family
        for spec in specs
        if spec.violation_family is not None
    } == set(VIOLATION_FAMILIES)


def test_room_overlap_candidate_creates_overlapping_room_activation() -> None:
    taxonomy = _taxonomy()
    controls = _controls(taxonomy)
    record = _record()
    spec = next(
        spec
        for spec in build_candidate_specs(record)
        if spec.violation_family == "room_overlap"
    )
    candidate = materialize_candidate(
        controls,
        controls,
        record,
        spec,
        taxonomy,
        footprint=None,
    )
    active = candidate.controls["room_logits"].sigmoid() > 0.5
    assert int(active[:, 2:7, 2:7].sum(dim=0).gt(1).sum()) > 0
    assert candidate.violation_labels["room_overlap"]


def test_outside_leakage_candidate_keeps_original_edit_box() -> None:
    taxonomy = _taxonomy()
    controls = _controls(taxonomy)
    record = _record()
    spec = next(
        spec
        for spec in build_candidate_specs(record)
        if spec.violation_family == "outside_leakage"
    )
    candidate = materialize_candidate(
        controls,
        controls,
        record,
        spec,
        taxonomy,
        footprint=np.ones((16, 16), dtype=bool),
    )
    delta = (candidate.controls["room_logits"] - controls["room_logits"]).abs()
    changed = delta.amax(dim=0) > 0
    x, y, width, height = candidate.bbox
    box = torch.zeros_like(changed)
    box[y : y + height, x : x + width] = True
    assert bool((changed & ~box).any())

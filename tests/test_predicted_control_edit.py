"""Tests for predicted-control instruction projection."""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.predicted_control_edit import (
    project_predicted_instruction,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_projected_instruction_changes_only_inside_declared_region() -> None:
    taxonomy = _taxonomy()
    before = torch.ones(16, 16, dtype=torch.long)
    logits = torch.full((len(taxonomy.room_ids), 16, 16), -4.0)
    logits[1] = 4.0
    controls = {
        "room_logits": logits,
        "program": torch.ones(len(taxonomy.room_ids), dtype=torch.bool),
    }
    after, mask = project_predicted_instruction(
        before,
        controls,
        (4, 4, 6, 6),
        (16, 16),
        taxonomy,
    )
    assert mask.sum() == 49
    assert torch.equal(after[~mask], before[~mask])
    assert torch.all(after[mask] == taxonomy.room_ids[1])


@pytest.mark.parametrize(
    ("operation", "family_name"),
    [
        ("opening_delete", "door"),
        ("furniture_delete", "furniture"),
    ],
)
def test_object_delete_removes_only_the_requested_family(
    operation: str,
    family_name: str,
) -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter({"Bed": 2}),
            "door": Counter({"Swing": 2}),
            "window": Counter({"Regular": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    before = torch.full((12, 12), taxonomy.room_ids[0], dtype=torch.long)
    family_id = (
        taxonomy.opening_ids[0]
        if family_name == "door"
        else taxonomy.furniture_ids[0]
    )
    before[4:7, 4:7] = family_id
    controls = {
        "room_logits": torch.zeros(
            len(taxonomy.room_ids),
            12,
            12,
        ),
        "program": torch.ones(len(taxonomy.room_ids), dtype=torch.bool),
    }
    after, mask = project_predicted_instruction(
        before,
        controls,
        (3, 3, 6, 6),
        (12, 12),
        taxonomy,
        operation=operation,
    )
    assert mask[4:7, 4:7].all()
    assert after[4:7, 4:7].eq(0).all()
    assert torch.equal(after[~mask], before[~mask])


def test_room_add_writes_only_dominant_new_room_target() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2, "Bath": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    before = torch.full(
        (12, 12),
        taxonomy.room_ids[0],
        dtype=torch.long,
    )
    edited = before.clone()
    edited[2:6, 2:7] = taxonomy.room_ids[1]
    edited[6:10, 5:10] = taxonomy.room_ids[2]
    controls = {
        "room_logits": torch.full(
            (len(taxonomy.room_ids), 12, 12),
            -8.0,
        ),
        "program": torch.ones(len(taxonomy.room_ids), dtype=torch.bool),
    }
    controls["room_logits"][0] = 4.0
    controls["room_logits"][1, 2:6, 2:7] = 8.0
    controls["room_logits"][2, 6:10, 5:10] = 8.0
    after, mask = project_predicted_instruction(
        before,
        controls,
        (0, 0, 6, 12),
        (12, 12),
        taxonomy,
        operation="room_add",
    )
    assert after[2:6, 2:7].eq(taxonomy.room_ids[1]).all()
    assert not after[6:10, 5:10].eq(taxonomy.room_ids[2]).any()
    assert torch.equal(after[~mask], before[~mask])


def test_room_add_keeps_only_largest_target_component() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    before = torch.full(
        (12, 12),
        taxonomy.room_ids[0],
        dtype=torch.long,
    )
    controls = {
        "room_logits": torch.full(
            (len(taxonomy.room_ids), 12, 12),
            -8.0,
        ),
        "program": torch.ones(len(taxonomy.room_ids), dtype=torch.bool),
    }
    controls["room_logits"][0] = 4.0
    controls["room_logits"][1, 1:6, 1:6] = 8.0
    controls["room_logits"][1, 8:10, 8:10] = 8.0
    after, _mask = project_predicted_instruction(
        before,
        controls,
        (1, 1, 9, 9),
        (12, 12),
        taxonomy,
        operation="room_add",
    )
    assert after[1:6, 1:6].eq(taxonomy.room_ids[1]).all()
    assert after[8:10, 8:10].eq(taxonomy.room_ids[0]).all()


def test_room_remove_merges_into_neighbor_class() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    before = torch.full(
        (12, 12),
        taxonomy.room_ids[1],
        dtype=torch.long,
    )
    before[:, :5] = taxonomy.room_ids[0]
    controls = {
        "room_logits": torch.full(
            (len(taxonomy.room_ids), 12, 12),
            -8.0,
        ),
        "program": torch.ones(len(taxonomy.room_ids), dtype=torch.bool),
    }
    controls["room_logits"][1] = 8.0
    after, _mask = project_predicted_instruction(
        before,
        controls,
        (0, 0, 6, 12),
        (12, 12),
        taxonomy,
        operation="room_remove",
    )
    assert after.eq(taxonomy.room_ids[1]).all()

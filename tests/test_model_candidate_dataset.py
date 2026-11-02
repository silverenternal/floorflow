"""Tests for model-origin candidate data and quality labels."""

from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.model_candidate_dataset import (
    control_quality,
    target_cache_path,
)
from floorflow.data.taxonomy import build_taxonomy


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 3, "Kitchen": 3}),
            "furniture": Counter({"Sink": 3}),
            "door": Counter({"Swing Beside": 3}),
            "window": Counter({"Regular": 3}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _controls(taxonomy):
    room = torch.full((len(taxonomy.room_ids), 16, 16), -8.0)
    room[0, 2:8, 2:8] = 8.0
    structure = torch.zeros(
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        16,
        16,
    )
    structure[len(taxonomy.room_ids), :, 0] = 1.0
    furniture = torch.zeros(
        len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
        16,
        16,
    )
    furniture[len(taxonomy.room_ids), 4:6, 4:6] = 1.0
    support = torch.ones(len(taxonomy.room_ids) + 4, 16, 16)
    program = torch.zeros(len(taxonomy.room_ids), dtype=torch.bool)
    program[0] = True
    return {
        "room_logits": room,
        "program": program,
        "structure": structure,
        "furniture": furniture,
        "support": support,
    }


def test_target_cache_path_maps_split_name() -> None:
    path = target_cache_path(
        "outputs/cache/cubicasa5k",
        "val",
        "high_quality_architectural__333__Floor-1",
    )
    assert str(path).endswith(
        "val/high_quality_architectural/333.npz"
    )


def test_identity_candidate_has_perfect_intent_and_target_quality() -> None:
    taxonomy = _taxonomy()
    controls = _controls(taxonomy)
    target = torch.zeros(16, 16, dtype=torch.long)
    target[2:8, 2:8] = taxonomy.room_ids[0]
    quality = control_quality(controls, controls, target, taxonomy)
    assert quality.room_intent_iou == 1.0
    assert quality.wall_intent_iou == 1.0
    assert quality.opening_intent_iou == 1.0
    assert quality.furniture_intent_iou == 1.0
    assert quality.target_room_iou == 1.0
    assert 0.0 <= quality.utility <= 1.0

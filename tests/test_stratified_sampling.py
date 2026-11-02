from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from floorflow.data.stratified_sampling import (
    FloorplanStratum,
    compute_floorplan_stratum,
    curriculum_sample_weights,
    stratified_loss_weights,
)
from floorflow.data.taxonomy import build_taxonomy


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Bed": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_floorplan_stratum_counts_room_and_objects() -> None:
    taxonomy = _taxonomy()
    target = np.zeros((16, 16), dtype=np.uint8)
    target[1:6, 1:6] = taxonomy.class_id("room:Bedroom")
    target[6:11, 6:11] = taxonomy.class_id("room:Kitchen")
    target[2, 2:4] = taxonomy.class_id("furniture:Bed")
    footprint = np.ones((16, 16), dtype=np.uint8)
    entry = compute_floorplan_stratum(
        "colorful/1__Floor-1",
        "train",
        target,
        footprint,
        taxonomy,
    )
    assert entry.corpus == "colorful"
    assert entry.room_class_count == 2
    assert entry.room_component_count == 2
    assert entry.furniture_pixels == 2
    assert entry.footprint_aspect == 1.0
    assert len(entry.present_room_classes) == 2
    assert taxonomy.class_id("room:Bedroom") in entry.present_room_classes
    assert taxonomy.class_id("furniture:Bed") in entry.present_furniture_classes
    assert entry.fine_stratum.startswith(entry.stratum)
    assert entry.instance_count_bin >= 0


def test_curriculum_weights_are_distribution_and_favor_late_complexity() -> None:
    base = FloorplanStratum(
        name="high_quality/1__Floor-1",
        split="train",
        corpus="high_quality",
        floor_id=1,
        room_class_count=1,
        room_component_count=1,
        rare_room_pixels=0,
        opening_pixels=0,
        furniture_pixels=0,
        footprint_pixels=100,
        footprint_fill_fraction=1.0,
        footprint_aspect=1.0,
        boundary_pixels=10,
        room_count_bin=0,
        object_count_bin=0,
        complexity_bin=0,
        has_rare=False,
        stratum="easy",
    )
    entries = [
        base,
        replace(
            base,
            name="colorful/2__Floor-1",
            room_class_count=10,
            room_component_count=12,
            object_count_bin=4,
            has_rare=True,
            stratum="hard",
        ),
    ]
    early = curriculum_sample_weights(entries, progress=0.0)
    late = curriculum_sample_weights(entries, progress=1.0)
    assert np.isclose(early.sum(), 1.0)
    assert late[1] > early[1]


def test_stratified_loss_weights_do_not_change_visitation() -> None:
    entries = [
        FloorplanStratum(
            name="a",
            split="train",
            corpus="x",
            floor_id=1,
            room_class_count=0,
            room_component_count=0,
            rare_room_pixels=0,
            opening_pixels=0,
            furniture_pixels=0,
            footprint_pixels=1,
            footprint_fill_fraction=1.0,
            footprint_aspect=1.0,
            boundary_pixels=1,
            room_count_bin=0,
            object_count_bin=0,
            complexity_bin=0,
            has_rare=True,
            stratum="a",
        ),
        FloorplanStratum(
            name="b",
            split="train",
            corpus="x",
            floor_id=1,
            room_class_count=1,
            room_component_count=1,
            rare_room_pixels=0,
            opening_pixels=0,
            furniture_pixels=0,
            footprint_pixels=1,
            footprint_fill_fraction=1.0,
            footprint_aspect=1.0,
            boundary_pixels=1,
            room_count_bin=0,
            object_count_bin=0,
            complexity_bin=0,
            has_rare=False,
            stratum="b",
        ),
    ]
    weights = stratified_loss_weights(entries)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[0] > weights[1]

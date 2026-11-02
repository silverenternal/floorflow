"""Tests for deterministic room-graph refinement candidates."""

from __future__ import annotations

from collections import Counter

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.room_graph_refinement import (
    graph_refinement_methods,
    refine_room_graph,
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


def test_refinement_changes_only_graph_and_preserves_identical_input() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[8:56, 8:31] = taxonomy.room_ids[0]
    labels[8:56, 33:56] = taxonomy.room_ids[1]
    wall = taxonomy.class_id("wall")
    labels[8:56, 8:11] = wall
    labels[8:56, 31:33] = wall
    labels[8:56, 53:56] = wall
    labels[8:11, 8:56] = wall
    labels[53:56, 8:56] = wall
    before = labels.copy()
    baseline = refine_room_graph(
        labels,
        taxonomy,
        footprint=labels > 0,
        method="wall_wall_primary",
    )
    refined = refine_room_graph(
        labels,
        taxonomy,
        footprint=labels > 0,
        method="direct_tiled",
    )
    assert len(baseline.nodes) >= len(refined.nodes)
    assert len(refined.nodes) == 2
    assert np.array_equal(labels, before)


def test_graph_refinement_methods_are_explicit() -> None:
    methods = graph_refinement_methods()
    assert "wall_wall_primary" in methods
    assert "direct_tiled" in methods
    assert "tiled_purity_050_wall_union" in methods

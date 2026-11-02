from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from floorflow.data.taxonomy import build_taxonomy
from floorflow.geometry.wall_graph_readout import (
    WallSkeletonPolicy,
    raster_labels_to_wall_plan,
)


def test_wall_skeleton_policy_validates_kernel_and_thresholds() -> None:
    WallSkeletonPolicy(close_kernel=0).validate()
    WallSkeletonPolicy(close_kernel=3).validate()

    with pytest.raises(ValueError, match="positive odd"):
        WallSkeletonPolicy(close_kernel=4).validate()
    with pytest.raises(ValueError, match="non-negative"):
        WallSkeletonPolicy(simplification_fraction=-0.1).validate()
    with pytest.raises(ValueError, match="non-negative"):
        WallSkeletonPolicy(minimum_segment_length=-1.0).validate()


def test_wall_skeleton_policy_minimum_length_filters_short_edges() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter({"Swing": 2}),
            "window": Counter(),
        },
        room_min_count=1,
    )
    wall_id = taxonomy.class_id("wall")
    labels = np.zeros((64, 64), dtype=np.int64)
    labels[16:48, 16:18] = wall_id

    permissive = raster_labels_to_wall_plan(
        labels,
        taxonomy,
        wall_policy=WallSkeletonPolicy(
            close_kernel=3,
            minimum_segment_length=1.0,
        ),
    )
    restrictive = raster_labels_to_wall_plan(
        labels,
        taxonomy,
        wall_policy=WallSkeletonPolicy(
            close_kernel=3,
            minimum_segment_length=100.0,
        ),
    )

    assert permissive.walls
    assert restrictive.walls == []

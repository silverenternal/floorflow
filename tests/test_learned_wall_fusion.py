"""Tests for bounded learned-wall endpoint fusion."""

import numpy as np
from shapely.geometry import Polygon

from floorflow.evaluation.canonical_floorplan import CanonicalWall
from floorflow.geometry.learned_wall_fusion import (
    fuse_learned_wall_endpoints,
)


def test_learned_fusion_moves_shared_junction_as_one_cluster() -> None:
    walls = (
        CanonicalWall("wall_01", (0, 4), (4, 4), 3.0),
        CanonicalWall("wall_02", (4, 4), (8, 4), 3.0),
    )
    learned = np.asarray(
        [
            [[0.0, 4.5], [4.0, 4.5]],
            [[4.0, 4.5], [8.0, 4.5]],
        ]
    )

    fused, edited = fuse_learned_wall_endpoints(
        walls,
        learned,
        np.ones(2),
        Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]),
    )

    assert edited == 3
    assert all(start[1] == 4.5 and end[1] == 4.5 for start, end in (
        (wall.start, wall.end) for wall in fused
    ))


def test_learned_fusion_is_bounded_and_requires_a_match() -> None:
    walls = (CanonicalWall("wall_01", (0, 4), (4, 4), 3.0),)
    footprint = Polygon([(0, 0), (8, 0), (8, 8), (0, 8)])

    near, edited = fuse_learned_wall_endpoints(
        walls,
        np.asarray([[[0.0, 7.0], [4.0, 7.0]]]),
        np.ones(1),
        footprint,
        maximum_residual=0.5,
    )
    far, far_edited = fuse_learned_wall_endpoints(
        walls,
        np.asarray([[[20.0, 20.0], [24.0, 20.0]]]),
        np.ones(1),
        footprint,
    )

    assert edited == 2
    assert abs(near[0].start[1] - 4.5) <= 1.0e-8
    assert far_edited == 0
    assert far == walls

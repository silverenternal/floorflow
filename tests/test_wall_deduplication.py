from __future__ import annotations

from floorflow.geometry.wall_deduplication import (
    is_near_duplicate_wall,
    prune_near_duplicate_walls,
)
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutWall,
)


def test_near_duplicate_wall_detection_requires_overlap_and_angle() -> None:
    main = ReadoutWall(start=(0.0, 10.0), end=(40.0, 10.0), thickness=3.0)
    shifted = ReadoutWall(start=(2.0, 11.0), end=(36.0, 11.0), thickness=3.0)
    perpendicular = ReadoutWall(
        start=(20.0, 10.0),
        end=(20.0, 30.0),
        thickness=3.0,
    )
    assert is_near_duplicate_wall(
        main,
        shifted,
        distance_tolerance=2.0,
        angle_tolerance=4.0,
        minimum_overlap_fraction=0.65,
    )
    assert not is_near_duplicate_wall(
        main,
        perpendicular,
        distance_tolerance=2.0,
        angle_tolerance=4.0,
        minimum_overlap_fraction=0.65,
    )


def test_near_duplicate_wall_pruning_keeps_nonneighbor_walls() -> None:
    plan = RasterWallPlan(
        frame=64.0,
        walls=[
            ReadoutWall(start=(0.0, 10.0), end=(40.0, 10.0), thickness=3.0),
            ReadoutWall(start=(2.0, 11.0), end=(36.0, 11.0), thickness=3.0),
            ReadoutWall(start=(0.0, 40.0), end=(40.0, 40.0), thickness=3.0),
        ],
        rooms=[],
        openings=[],
    )
    pruned = prune_near_duplicate_walls(
        plan,
        distance_tolerance_fraction=0.04,
    )
    assert len(pruned.walls) == 2


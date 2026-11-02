import math
from collections import Counter

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.geometry.reconciliation import fuse_wall_plans, reconcile_plan
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutWall,
    raster_labels_to_wall_plan,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Kitchen": 4}),
            "furniture": Counter({"Sink": 4}),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _fixture():
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    wall = taxonomy.class_id("wall")
    labels[12:52, 12:30] = taxonomy.room_ids[0]
    labels[12:52, 34:52] = taxonomy.room_ids[1]
    labels[8:56, 8:12] = wall
    labels[8:56, 52:56] = wall
    labels[8:12, 8:56] = wall
    labels[52:56, 8:56] = wall
    labels[8:56, 30:34] = wall
    labels[28:36, 52:56] = taxonomy.opening_ids[0]
    return taxonomy, labels


def test_reconcile_removes_duplicate_wall_and_projects_opening() -> None:
    taxonomy, labels = _fixture()
    plan = raster_labels_to_wall_plan(labels, taxonomy)
    duplicate = plan.walls[0]
    plan.walls.append(
        ReadoutWall(
            start=duplicate.end,
            end=duplicate.start,
            thickness=duplicate.thickness,
        )
    )
    reconciled = reconcile_plan(plan, labels, taxonomy)
    assert len(reconciled.walls) == len(plan.walls) - 1
    assert reconciled.openings


def test_ink_gated_fusion_is_deterministic_and_non_degrading() -> None:
    taxonomy, labels = _fixture()
    primary = raster_labels_to_wall_plan(labels, taxonomy)
    donor = raster_labels_to_wall_plan(labels, taxonomy)
    donor.walls.append(
        ReadoutWall(
            start=(16.0, 32.0),
            end=(48.0, 32.0),
            thickness=3.0,
        )
    )
    ink = np.zeros_like(labels, dtype=np.uint8)
    ink[31:34, 16:49] = 1
    first = fuse_wall_plans(primary, donor, support_mask=ink)
    second = fuse_wall_plans(primary, donor, support_mask=ink)
    assert first.to_json() == second.to_json()
    assert len(first.walls) == len(primary.walls) + 1
    report = fpeval_raster_metrics(labels, labels, taxonomy)
    assert report["prediction_validity"]["watertight"]


def test_gap_closing_bridges_small_gap_without_splitting_walls() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    plan = RasterWallPlan(
        frame=64.0,
        walls=[
            ReadoutWall(start=(10.0, 20.0), end=(30.0, 20.0), thickness=3.0),
            ReadoutWall(start=(31.2, 20.0), end=(50.0, 20.0), thickness=3.0),
        ],
        rooms=[],
        openings=[],
    )
    closed = reconcile_plan(plan, labels, taxonomy, tolerance_fraction=0.01)
    assert len(closed.walls) == 3
    assert any(
        math.dist(wall.start, wall.end) <= 1.5
        for wall in closed.walls
    )


def test_gap_closing_ignores_large_gap() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    plan = RasterWallPlan(
        frame=64.0,
        walls=[
            ReadoutWall(start=(10.0, 20.0), end=(20.0, 20.0), thickness=3.0),
            ReadoutWall(start=(30.0, 20.0), end=(40.0, 20.0), thickness=3.0),
        ],
        rooms=[],
        openings=[],
    )
    closed = reconcile_plan(
        plan,
        labels,
        taxonomy,
        tolerance_fraction=0.01,
        gap_tolerance_fraction=0.02,
    )
    assert len(closed.walls) == 2


def test_open_endpoint_repair_closes_frame_boundary_walls() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    plan = RasterWallPlan(
        frame=64.0,
        walls=[
            ReadoutWall(start=(0.0, 20.0), end=(64.0, 20.0), thickness=3.0),
            ReadoutWall(start=(0.0, 40.0), end=(64.0, 40.0), thickness=3.0),
        ],
        rooms=[],
        openings=[],
    )
    repaired = reconcile_plan(
        plan,
        labels,
        taxonomy,
        tolerance_fraction=0.01,
        repair_open_endpoints=True,
    )
    assert len(repaired.walls) == 4
    assert sum(wall.wall_type == "exterior" for wall in repaired.walls) == 2


def test_open_endpoint_repair_prunes_short_open_spur() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    plan = RasterWallPlan(
        frame=64.0,
        walls=[
            ReadoutWall(start=(10.0, 20.0), end=(50.0, 20.0), thickness=3.0),
            ReadoutWall(start=(10.0, 20.0), end=(11.0, 20.0), thickness=3.0),
        ],
        rooms=[],
        openings=[],
    )
    repaired = reconcile_plan(
        plan,
        labels,
        taxonomy,
        tolerance_fraction=0.01,
        repair_open_endpoints=True,
    )
    assert len(repaired.walls) == 1

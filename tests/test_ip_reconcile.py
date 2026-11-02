from collections import Counter

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.geometry.ip_reconcile import ip_reconcile_plan
from floorflow.geometry.reconciliation import reconcile_plan
from floorflow.geometry.wall_graph_readout import (
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


def test_ip_reconcile_removes_duplicate_and_keeps_openings() -> None:
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
    reconciled = ip_reconcile_plan(
        plan,
        labels,
        taxonomy,
        use_milp=False,
    )
    assert len(reconciled.walls) == len(plan.walls) - 1
    assert reconciled.openings
    assert reconciled.rooms
    assert reconciled.room_source in {"ip_reconciled_faces", plan.room_source}


def test_ip_reconcile_merges_collinear_fragments() -> None:
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
    plan = raster_labels_to_wall_plan(labels, taxonomy)
    top = [w for w in plan.walls if abs(w.start[1] - 10.0) < 3.0 or abs(w.end[1] - 10.0) < 3.0]
    reconciled = ip_reconcile_plan(plan, labels, taxonomy)
    assert reconciled.walls
    assert len(reconciled.walls) <= len(plan.walls)
    assert all(w.start != w.end for w in reconciled.walls)
    assert top


def test_reconcile_plan_dispatches_to_ip() -> None:
    taxonomy, labels = _fixture()
    plan = raster_labels_to_wall_plan(labels, taxonomy)
    shapely = reconcile_plan(plan, labels, taxonomy)
    ip = reconcile_plan(plan, labels, taxonomy, use_ip=True)
    assert shapely.walls and ip.walls
    labels = {room["label"] for room in ip.rooms}
    assert labels >= {"room:Bedroom", "room:Kitchen"}

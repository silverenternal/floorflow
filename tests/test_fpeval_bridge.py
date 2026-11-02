from collections import Counter
import json

import numpy as np

from floorflow.evaluation.fpeval_bridge import (
    count_double_walls,
    fpeval_raster_metrics,
    to_fpeval_plan,
)
from floorflow.data.taxonomy import build_taxonomy
from floorflow.geometry.wall_graph_readout import ReadoutWall, raster_labels_to_wall_plan


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


def test_raster_reading_produces_walls_rooms_and_openings() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((96, 96), dtype=np.int64)
    wall = taxonomy.class_id("wall")
    labels[20:80, 12:42] = taxonomy.room_ids[0]
    labels[20:80, 46:84] = taxonomy.room_ids[1]
    labels[8:88, 8:12] = wall
    labels[8:88, 84:88] = wall
    labels[8:12, 8:88] = wall
    labels[84:88, 8:88] = wall
    labels[8:88, 42:46] = wall
    labels[40:48, 84:88] = taxonomy.opening_ids[0]
    plan = raster_labels_to_wall_plan(labels, taxonomy)
    assert len(plan.walls) >= 3
    assert len(plan.rooms) >= 1
    assert {room["label"] for room in plan.rooms} == {
        "room:Bedroom",
        "room:Kitchen",
    }
    assert plan.openings
    encoded = plan.to_json()
    assert encoded["walls"] and encoded["rooms"]


def test_fpeval_bridge_returns_tolerance_reports() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    wall = taxonomy.class_id("wall")
    labels[8:56, 8:12] = wall
    labels[8:56, 52:56] = wall
    labels[8:12, 8:56] = wall
    labels[52:56, 8:56] = wall
    labels[12:52, 12:52] = taxonomy.room_ids[0]
    report = fpeval_raster_metrics(
        labels,
        labels,
        taxonomy,
    )
    assert set(report["tolerances"]) == {
        "0.015",
        "0.05",
    }
    for summary in report["summary"].values():
        assert summary["wall_f1"] == 1.0
        assert summary["room_f1"] == 1.0
        assert summary["room_label_accuracy"] == 1.0
        assert summary["edit_cost"] == 0.0
    assert report["prediction_validity"]["valid"]
    assert report["prediction_validity"]["watertight"]
    assert report["target_validity"]["valid"]
    assert report["prediction_double_walls"] == 0
    json.dumps(report)


def test_double_wall_counter_is_orientation_invariant() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    wall = taxonomy.class_id("wall")
    labels[8:56, 8:12] = wall
    labels[8:56, 52:56] = wall
    labels[8:12, 8:56] = wall
    labels[52:56, 8:56] = wall
    labels[12:52, 12:52] = taxonomy.room_ids[0]
    raster_plan = raster_labels_to_wall_plan(labels, taxonomy)
    raster_plan.walls.append(
        ReadoutWall(
            start=raster_plan.walls[0].end,
            end=raster_plan.walls[0].start,
            thickness=raster_plan.walls[0].thickness,
        )
    )
    assert count_double_walls(to_fpeval_plan(raster_plan)) == 1


def test_fpeval_bridge_accepts_reconcile_and_opening_heatmap() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    wall = taxonomy.class_id("wall")
    labels[8:56, 8:12] = wall
    labels[8:56, 52:56] = wall
    labels[8:12, 8:56] = wall
    labels[52:56, 8:56] = wall
    labels[12:52, 12:52] = taxonomy.room_ids[0]
    opening = taxonomy.opening_ids[0]
    labels[54:56, 24:34] = opening
    foreground = np.zeros(labels.shape, dtype=np.float32)
    foreground[54:56, 24:34] = 0.9
    logits = np.zeros((1, *labels.shape), dtype=np.float32)
    logits[0, 52:58, 22:36] = 2.0
    report = fpeval_raster_metrics(
        labels,
        labels,
        taxonomy,
        reconcile=True,
        prediction_opening_probability=foreground,
        prediction_opening_logits=logits,
        prediction_opening_threshold=0.5,
    )
    assert report["reconciled"] is True
    assert report["opening_readout_source"] in {"heatmap_interval", "labels"}
    assert report["prediction_validity"]["valid"]

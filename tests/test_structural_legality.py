from __future__ import annotations

from collections import Counter

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.structural_legality import (
    structural_legality_metrics,
    wall_junction_summary,
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


def _floorplan(taxonomy):
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[8:56, 8:31] = taxonomy.room_ids[0]
    labels[8:56, 33:56] = taxonomy.room_ids[1]
    wall = taxonomy.class_id("wall")
    labels[8:56, 8:11] = wall
    labels[8:56, 31:33] = wall
    labels[8:56, 53:56] = wall
    labels[8:11, 8:56] = wall
    labels[53:56, 8:56] = wall
    return labels


def test_structural_metrics_are_perfect_for_identical_plans() -> None:
    taxonomy = _taxonomy()
    labels = _floorplan(taxonomy)
    metrics = structural_legality_metrics(labels, labels, taxonomy)
    assert metrics["parsable"] == 1.0
    assert metrics["room_closure_rate"] == 1.0
    assert metrics["self_intersection_rate"] == 0.0
    assert metrics["room_overlap_ratio"] == 0.0
    assert metrics["out_of_boundary_ratio"] == 0.0
    assert metrics["node_f1"] == 1.0
    assert metrics["edge_overlap"] == 1.0
    assert metrics["ged"] == 0.0


def test_wall_junction_summary_reports_continuity_and_degrees() -> None:
    class Wall:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    summary = wall_junction_summary(
        [
            Wall((0.0, 0.0), (10.0, 0.0)),
            Wall((10.0, 0.0), (10.0, 10.0)),
        ]
    )
    assert summary["junction_count"] == 3
    assert summary["open_endpoint_count"] == 2
    assert summary["degree_ge_2_rate"] == 1.0 / 3.0
    assert summary["degree_ge_3_rate"] == 0.0

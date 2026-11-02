from __future__ import annotations

from collections import Counter

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.fevp_metric_contracts import (
    FEVPTolerances,
    edge_precision_recall_f1,
    node_precision_recall_f1,
    outside_layer_metrics,
    room_instances,
    validity_checks,
    watertight_diagnostics,
)
from floorflow.evaluation.housemind_structure import RoomTypeGraph
from floorflow.geometry.wall_graph_readout import (
    ReadoutWall,
    raster_labels_to_wall_plan,
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


def _labels(taxonomy):
    labels = np.zeros((64, 64), dtype=np.uint8)
    wall = taxonomy.class_id("wall")
    bedroom = taxonomy.class_id("room:Bedroom")
    kitchen = taxonomy.class_id("room:Kitchen")
    labels[8:56, 8:31] = bedroom
    labels[8:56, 33:56] = kitchen
    labels[8:56, 8:11] = wall
    labels[8:56, 31:33] = wall
    labels[8:56, 53:56] = wall
    labels[8:11, 8:56] = wall
    labels[53:56, 8:56] = wall
    return labels


def _outside(before, after, changed, region):
    taxonomy = _taxonomy()
    return outside_layer_metrics(
        before,
        after,
        changed,
        region,
        before > 0,
        taxonomy,
        tolerances=FEVPTolerances(),
    )


def test_outside_layers_accept_exact_unaffected_rooms() -> None:
    taxonomy = _taxonomy()
    before = _labels(taxonomy)
    after = before.copy()
    changed = np.zeros_like(before, dtype=bool)
    region = np.zeros_like(changed)
    region[12:48, 12:28] = True
    changed[20:30, 15:25] = True
    metrics = _outside(before, after, changed, region)
    assert metrics["outside_pixel_exact"] == 1.0
    assert metrics["outside_room_level_exact"] == 1.0
    assert metrics["outside_edge_exact"] == 1.0
    assert metrics["locality_violation_rate"] == 0.0


def test_outside_room_layer_rejects_one_pixel_move_and_label_change() -> None:
    taxonomy = _taxonomy()
    before = _labels(taxonomy)
    region = np.zeros_like(before, dtype=bool)
    changed = np.zeros_like(region)
    region[12:48, 12:28] = True
    changed[20:30, 15:25] = True

    moved = before.copy()
    kitchen = taxonomy.class_id("room:Kitchen")
    wall = taxonomy.class_id("wall")
    moved[8:56, 33] = wall
    moved[8:56, 56] = kitchen
    moved_metrics = _outside(before, moved, changed, region)
    assert moved_metrics["outside_room_level_exact"] == 0.0

    relabeled = before.copy()
    rare = taxonomy.class_id("room_rare")
    relabeled[relabeled == kitchen] = rare
    label_metrics = _outside(before, relabeled, changed, region)
    assert label_metrics["outside_room_label_exact"] == 0.0


def test_validity_checks_conjoin_five_decomposed_checks() -> None:
    taxonomy = _taxonomy()
    labels = _labels(taxonomy)
    plan = raster_labels_to_wall_plan(labels, taxonomy)
    checks = validity_checks(
        plan,
        labels > 0,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    assert checks["validity"] == 1.0
    assert all(
        checks[key] == 1.0
        for key in (
            "outer_boundary_closed",
            "room_polygon_closed",
            "no_self_intersection",
            "no_room_overlap",
            "semantic_complete",
        )
    )


def test_watertight_diagnostics_detect_hole_and_dangling_edges() -> None:
    taxonomy = _taxonomy()
    labels = _labels(taxonomy)
    labels[20:22, 20:22] = 0
    plan = raster_labels_to_wall_plan(labels, taxonomy)
    plan.walls.append(ReadoutWall((1.0, 1.0), (10.0, 1.0), 3.0))
    diagnostics = watertight_diagnostics(
        plan,
        taxonomy=taxonomy,
        tolerances=FEVPTolerances(),
        fpeval_watertight=False,
        labels=labels,
        footprint=np.ones_like(labels, dtype=bool),
    )
    assert diagnostics["watertight"] is False
    assert diagnostics["hole_count"] >= 1
    assert diagnostics["dangling_endpoint_rate"] > 0.0
    assert "fpeval_not_watertight" in diagnostics["watertight_issue_codes"]


def test_graph_metrics_decompose_precision_recall_and_f1() -> None:
    predicted = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Kitchen", "Bath"}),
        edges=frozenset({("Bedroom", "Kitchen"), ("Kitchen", "Bath")}),
    )
    target = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Kitchen"}),
        edges=frozenset({("Bedroom", "Kitchen")}),
    )
    nodes = node_precision_recall_f1(predicted, target)
    edges = edge_precision_recall_f1(predicted, target)
    assert nodes["precision"] == 2.0 / 3.0
    assert nodes["recall"] == 1.0
    assert edges["precision"] == 0.5
    assert edges["recall"] == 1.0
    assert edges["f1"] == 2.0 / 3.0


def test_room_instances_are_one_to_one_not_semantic_sets() -> None:
    taxonomy = _taxonomy()
    labels = _labels(taxonomy)
    bedroom = taxonomy.class_id("room:Bedroom")
    labels[15:45, 36:52] = bedroom
    instances = room_instances(
        labels,
        taxonomy,
        min_area=FEVPTolerances().room_min_area_px,
    )
    bedroom_instances = [
        instance for instance in instances if instance.label == "room:Bedroom"
    ]
    assert len(bedroom_instances) >= 2

from __future__ import annotations

import inspect

import numpy as np
from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.generation_plausibility import (
    generation_plausibility,
    plan_generation_plausibility,
    raster_generation_plausibility,
)
from floorflow.geometry.partition_polygons import (
    explicit_partition_plan,
    partition_polygon_rows,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "room:LivingRoom",
            "room:Bedroom",
            "door",
        ),
        room_map={"LivingRoom": "room:LivingRoom", "Bedroom": "room:Bedroom"},
        furniture_map={},
        door_map={"Swing": "door"},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def _tiled_plan() -> tuple[np.ndarray, np.ndarray, FloorplanTaxonomy]:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    footprint = np.zeros_like(labels)
    footprint[4:60, 4:60] = 1
    labels[4:60, 4:32] = taxonomy.class_id("room:LivingRoom")
    labels[4:60, 32:60] = taxonomy.class_id("room:Bedroom")
    labels[4:60, 31:33] = taxonomy.class_id("wall")
    labels[4:60, 3:6] = taxonomy.class_id("wall")
    labels[4:60, 58:61] = taxonomy.class_id("wall")
    labels[3:6, 3:61] = taxonomy.class_id("wall")
    labels[58:61, 3:61] = taxonomy.class_id("wall")
    labels[28:31, 31:33] = taxonomy.class_id("door")
    labels[~footprint.astype(bool)] = 0
    return labels, footprint, taxonomy


def test_reconciliation_ip_switch_defaults_to_false() -> None:
    assert (
        inspect.signature(generation_plausibility)
        .parameters["reconcile_use_ip"]
        .default
        is False
    )


def test_raster_plausibility_accepts_complete_partition() -> None:
    labels, footprint, taxonomy = _tiled_plan()
    metrics = raster_generation_plausibility(labels, footprint, taxonomy)
    assert metrics["room_coverage"] > 0.9
    assert metrics["room_area_fraction_of_interior"] > 0.8
    assert metrics["outside_footprint_fraction"] == 0.0
    assert metrics["raster_componentwise_wall_closure_ok"]
    assert metrics["wall_closure"] > 0.9


def test_plan_plausibility_reports_parse_and_topology_fields() -> None:
    labels, footprint, taxonomy = _tiled_plan()
    metrics = plan_generation_plausibility(labels, footprint, taxonomy)
    assert metrics["parsable"] == 1.0
    assert metrics["room_polygon_closed"] == 1.0
    assert metrics["wall_count"] > 0
    assert metrics["room_face_count"] > 0
    assert 0.0 <= metrics["room_face_retention"] <= 1.0
    assert metrics["functional_watertight"] in {0.0, 1.0}
    assert isinstance(metrics["plan_failure_codes"], list)


def test_generation_plausibility_detects_outside_labels() -> None:
    labels, footprint, taxonomy = _tiled_plan()
    labels[0, 0] = taxonomy.class_id("room:LivingRoom")
    metrics = generation_plausibility(labels, footprint, taxonomy)
    assert metrics["outside_footprint_fraction"] > 0.0
    assert not metrics["raster_outside_footprint_ok"]
    assert "outside_footprint" in metrics["raster_failure_codes"]


def test_functional_watertight_rejects_room_face_collapse() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    footprint = np.zeros_like(labels)
    footprint[4:60, 4:60] = 1
    labels[7:57, 7:57] = taxonomy.class_id("room:LivingRoom")
    labels[7:57, 32:57] = taxonomy.class_id("room:Bedroom")
    labels[4:60, 4:8] = taxonomy.class_id("wall")
    labels[4:60, 56:60] = taxonomy.class_id("wall")
    labels[4:8, 4:60] = taxonomy.class_id("wall")
    labels[56:60, 4:60] = taxonomy.class_id("wall")
    metrics = plan_generation_plausibility(labels, footprint, taxonomy)
    assert metrics["room_face_retention"] <= 0.5
    assert metrics["functional_watertight"] == 0.0
    assert "room_face_collapse" in metrics["plan_failure_codes"]


def test_partition_polygon_readout_preserves_room_identity() -> None:
    labels, footprint, taxonomy = _tiled_plan()
    rooms = partition_polygon_rows(labels, footprint, taxonomy)
    plan = explicit_partition_plan(labels, footprint, taxonomy)
    assert len(rooms) == 2
    assert plan.room_source == "explicit_partition"
    polygons = [Polygon(row["polygon"]) for row in plan.rooms]
    assert all(polygon.is_valid for polygon in polygons)
    assert polygons[0].intersection(polygons[1]).area == 0.0
    assert polygons[0].union(polygons[1]).area == footprint.sum()


def test_explicit_partition_reports_functional_watertight_with_nested_room() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((96, 96), dtype=np.int64)
    footprint = np.zeros_like(labels)
    footprint[4:92, 4:92] = 1
    labels[8:88, 8:88] = taxonomy.class_id("room:LivingRoom")
    labels[36:60, 36:60] = taxonomy.class_id("room:Bedroom")
    labels[4:92, 4:8] = taxonomy.class_id("wall")
    labels[4:92, 88:92] = taxonomy.class_id("wall")
    labels[4:8, 4:92] = taxonomy.class_id("wall")
    labels[88:92, 4:92] = taxonomy.class_id("wall")
    metrics = generation_plausibility(
        labels,
        footprint,
        taxonomy,
        explicit_rooms=True,
    )
    assert metrics["room_face_retention"] == 1.0
    assert metrics["no_room_overlap"] == 1.0
    assert metrics["functional_watertight"] == 1.0


def test_componentwise_wall_closure_handles_disconnected_footprints() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    footprint = np.zeros_like(labels)
    footprint[4:28, 4:28] = 1
    footprint[36:60, 36:60] = 1
    labels[4:28, 4:8] = taxonomy.class_id("wall")
    labels[4:28, 24:28] = taxonomy.class_id("wall")
    labels[4:8, 4:28] = taxonomy.class_id("wall")
    labels[24:28, 4:28] = taxonomy.class_id("wall")
    labels[36:60, 36:40] = taxonomy.class_id("wall")
    labels[36:60, 56:60] = taxonomy.class_id("wall")
    labels[36:40, 36:60] = taxonomy.class_id("wall")
    labels[56:60, 36:60] = taxonomy.class_id("wall")
    labels[8:24, 8:24] = taxonomy.class_id("room:LivingRoom")
    labels[40:56, 40:56] = taxonomy.class_id("room:Bedroom")
    metrics = raster_generation_plausibility(labels, footprint, taxonomy)
    assert metrics["wall_closure"] < 0.6
    assert metrics["componentwise_wall_closure"] == 1.0
    assert metrics["raster_componentwise_wall_closure_ok"]


def test_componentwise_topology_ignores_subroom_footprint_specks() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    footprint = np.zeros_like(labels)
    footprint[4:56, 4:56] = 1
    footprint[0:2, 0:2] = 1
    labels[8:52, 8:52] = taxonomy.class_id("room:LivingRoom")
    labels[8:52, 4:8] = taxonomy.class_id("wall")
    labels[8:52, 52:56] = taxonomy.class_id("wall")
    labels[4:56, 4:56] = np.where(
        labels[4:56, 4:56] == 0,
        taxonomy.class_id("wall"),
        labels[4:56, 4:56],
    )
    metrics = generation_plausibility(
        labels,
        footprint,
        taxonomy,
        explicit_rooms=True,
    )
    assert metrics["footprint_component_count"] == 2
    assert metrics["exportable_footprint_component_count"] == 1
    assert metrics["functional_watertight"] == 1.0



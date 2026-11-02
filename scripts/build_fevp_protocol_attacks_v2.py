#!/usr/bin/env python3
"""Run deterministic negative controls for every mandatory FEVP metric."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.fevp_metric_contracts import (
    FEVPTolerances,
    edge_precision_recall_f1,
    node_precision_recall_f1,
    outside_layer_metrics,
    validity_checks,
    watertight_diagnostics,
)
from floorflow.evaluation.housemind_structure import (
    RoomTypeGraph,
    housemind_style_ged,
)
from floorflow.geometry.wall_graph_readout import (
    ReadoutWall,
    raster_labels_to_wall_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/fevp_metric_contract_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_protocol_attacks_v2.json",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_protocol_attack_rows_v2.json",
    )
    parser.add_argument(
        "--examples-output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_metric_contract_examples_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_protocol_attacks_v2.md",
    )
    parser.add_argument(
        "--case-schema",
        type=Path,
        default=ROOT / "schema/fevp_case_row_v2.schema.json",
    )
    return parser.parse_args()


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


def _outside_context():
    taxonomy = _taxonomy()
    before = _labels(taxonomy)
    region = np.zeros_like(before, dtype=bool)
    changed = np.zeros_like(region)
    region[12:48, 12:28] = True
    changed[20:30, 15:25] = True
    return taxonomy, before, region, changed


def _edge_outside_context():
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.uint8)
    wall = taxonomy.class_id("wall")
    bedroom = taxonomy.class_id("room:Bedroom")
    kitchen = taxonomy.class_id("room:Kitchen")
    labels[8:56, 5:18] = bedroom
    labels[8:56, 20:35] = kitchen
    labels[8:56, 35:50] = bedroom
    labels[8:56, 5] = wall
    labels[8:56, 18:20] = wall
    labels[8:56, 50] = wall
    labels[8, 5:51] = wall
    labels[55, 5:51] = wall
    region = np.zeros_like(labels, dtype=bool)
    changed = np.zeros_like(region)
    region[12:50, 7:16] = True
    changed[20:30, 8:14] = True
    return taxonomy, labels, region, changed


def _outside(
    metric_name: str,
    label: str,
    after: np.ndarray,
    *,
    direction: str = "decrease",
) -> dict:
    taxonomy, before, region, changed = _outside_context()
    baseline = outside_layer_metrics(
        before,
        before,
        changed,
        region,
        before > 0,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    attacked = outside_layer_metrics(
        before,
        after,
        changed,
        region,
        before > 0,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    return _attack_row(
        "outside_exact",
        label,
        metric_name,
        baseline[metric_name],
        attacked[metric_name],
        direction=direction,
    )


def _attack_row(
    metric: str,
    attack: str,
    submetric: str,
    baseline,
    attacked,
    *,
    direction: str = "decrease",
) -> dict:
    baseline_value = float(baseline)
    attacked_value = float(attacked)
    sensitive = (
        attacked_value < baseline_value
        if direction == "decrease"
        else attacked_value > baseline_value
    )
    return {
        "metric": metric,
        "attack": attack,
        "submetric": submetric,
        "baseline": baseline_value,
        "attacked": attacked_value,
        "delta": attacked_value - baseline_value,
        "direction": direction,
        "sensitive": bool(sensitive),
        "false_pass": bool(not sensitive),
        "false_fail": bool(
            baseline_value < 0.999
            if direction == "decrease"
            else baseline_value > 0.001
        ),
    }


def outside_attacks() -> list[dict]:
    taxonomy, before, _region, _changed = _outside_context()
    kitchen = taxonomy.class_id("room:Kitchen")
    rare = taxonomy.class_id("room_rare")
    moved = before.copy()
    moved[8:56, 33:56] = 0
    moved[8:56, 34:57] = kitchen
    relabeled = before.copy()
    relabeled[relabeled == kitchen] = rare
    edge_taxonomy, edge_before, edge_region, edge_changed = (
        _edge_outside_context()
    )
    shifted = edge_before.copy()
    shifted[8:56, 35:50] = 0
    shifted[8:56, 39:54] = edge_taxonomy.class_id("room:Bedroom")
    edge_baseline = outside_layer_metrics(
        edge_before,
        edge_before,
        edge_changed,
        edge_region,
        edge_before > 0,
        edge_taxonomy,
        tolerances=FEVPTolerances(),
    )
    edge_attacked = outside_layer_metrics(
        edge_before,
        shifted,
        edge_changed,
        edge_region,
        edge_before > 0,
        edge_taxonomy,
        tolerances=FEVPTolerances(),
    )
    return [
        _outside("outside_room_level_exact", "move_untouched_room_1px", moved),
        _outside("outside_room_label_exact", "change_untouched_room_label", relabeled),
        _attack_row(
            "outside_exact",
            "shift_untouched_room_beyond_2px",
            "outside_edge_exact",
            edge_baseline["outside_edge_exact"],
            edge_attacked["outside_edge_exact"],
        ),
        _outside(
            "locality_violation_rate",
            "inject_outside_pixel_change",
            moved,
            direction="increase",
        ),
    ]


def validity_attacks() -> list[dict]:
    taxonomy = _taxonomy()
    labels = _labels(taxonomy)
    base = raster_labels_to_wall_plan(labels, taxonomy)
    footprint = labels > 0
    baseline = validity_checks(
        base,
        footprint,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    rows = []

    open_outer = validity_checks(
        base,
        np.zeros_like(footprint),
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    rows.append(
        _attack_row(
            "validity",
            "open_outer_boundary",
            "outer_boundary_closed",
            baseline["outer_boundary_closed"],
            open_outer["outer_boundary_closed"],
        )
    )
    open_room_plan = copy.deepcopy(base)
    open_room_plan.rooms[0]["polygon"] = []
    open_room = validity_checks(
        open_room_plan,
        footprint,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    rows.append(
        _attack_row(
            "validity",
            "open_room_polygon",
            "room_polygon_closed",
            baseline["room_polygon_closed"],
            open_room["room_polygon_closed"],
        )
    )
    bowtie_plan = copy.deepcopy(base)
    bowtie_plan.rooms[0]["polygon"] = [
        [8, 8],
        [30, 55],
        [30, 8],
        [8, 55],
        [8, 8],
    ]
    bowtie = validity_checks(
        bowtie_plan,
        footprint,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    rows.append(
        _attack_row(
            "validity",
            "self_intersecting_room",
            "no_self_intersection",
            baseline["no_self_intersection"],
            bowtie["no_self_intersection"],
        )
    )
    overlap_plan = copy.deepcopy(base)
    overlap_plan.rooms[1]["polygon"] = copy.deepcopy(
        overlap_plan.rooms[0]["polygon"]
    )
    overlap = validity_checks(
        overlap_plan,
        footprint,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    rows.append(
        _attack_row(
            "validity",
            "overlap_two_rooms",
            "no_room_overlap",
            baseline["no_room_overlap"],
            overlap["no_room_overlap"],
        )
    )
    illegal_plan = copy.deepcopy(base)
    illegal_plan.rooms[0]["label"] = "room:NotLegal"
    illegal = validity_checks(
        illegal_plan,
        footprint,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    rows.append(
        _attack_row(
            "validity",
            "illegal_room_label",
            "semantic_complete",
            baseline["semantic_complete"],
            illegal["semantic_complete"],
        )
    )
    rows.append(
        _attack_row(
            "validity",
            "joint_validity_after_self_intersection",
            "validity",
            baseline["validity"],
            bowtie["validity"],
        )
    )
    return rows


def watertight_attacks() -> list[dict]:
    taxonomy = _taxonomy()
    labels = _labels(taxonomy)
    footprint = np.ones_like(labels, dtype=bool)
    plan = raster_labels_to_wall_plan(labels, taxonomy)
    baseline = watertight_diagnostics(
        plan,
        taxonomy=taxonomy,
        labels=labels,
        footprint=footprint,
        tolerances=FEVPTolerances(),
        fpeval_watertight=True,
    )
    hole_labels = labels.copy()
    hole_labels[20:21, 20:21] = 0
    hole = watertight_diagnostics(
        plan,
        taxonomy=taxonomy,
        labels=hole_labels,
        footprint=footprint,
        tolerances=FEVPTolerances(),
        fpeval_watertight=True,
    )
    disconnected_plan = copy.deepcopy(plan)
    disconnected_plan.rooms[1]["polygon"] = [
        [point[0] + 100, point[1] + 100]
        for point in disconnected_plan.rooms[1]["polygon"]
    ]
    disconnected = watertight_diagnostics(
        disconnected_plan,
        taxonomy=taxonomy,
        labels=labels,
        footprint=footprint,
        tolerances=FEVPTolerances(),
    )
    dangling_plan = copy.deepcopy(plan)
    dangling_plan.walls.append(ReadoutWall((1.0, 1.0), (10.0, 1.0), 3.0))
    dangling = watertight_diagnostics(
        dangling_plan,
        taxonomy=taxonomy,
        labels=labels,
        footprint=footprint,
        tolerances=FEVPTolerances(),
    )
    return [
        _attack_row(
            "conditional_watertightness",
            "inject_one_pixel_hole",
            "hole_count",
            baseline["hole_count"],
            hole["hole_count"],
            direction="increase",
        ),
        _attack_row(
            "conditional_watertightness",
            "disconnect_room_component",
            "disconnected_component_count",
            baseline["disconnected_component_count"],
            disconnected["disconnected_component_count"],
            direction="increase",
        ),
        _attack_row(
            "conditional_watertightness",
            "open_dangling_wall",
            "dangling_endpoint_rate",
            baseline["dangling_endpoint_rate"],
            dangling["dangling_endpoint_rate"],
            direction="increase",
        ),
        _attack_row(
            "conditional_watertightness",
            "one_pixel_hole_watertight_rate",
            "watertight_rate",
            baseline["watertight_rate"],
            hole["watertight_rate"],
        ),
    ]


def graph_attacks() -> list[dict]:
    target = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Kitchen"}),
        edges=frozenset({("Bedroom", "Kitchen")}),
    )
    baseline_nodes = node_precision_recall_f1(target, target)
    baseline_edges = edge_precision_recall_f1(target, target)
    deleted_node = RoomTypeGraph(
        labels=frozenset({"Bedroom"}),
        edges=frozenset(),
    )
    extra_node = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Kitchen", "Bath"}),
        edges=frozenset({("Bedroom", "Kitchen")}),
    )
    relabeled_node = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Bath"}),
        edges=frozenset({("Bedroom", "Bath")}),
    )
    deleted_edge = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Kitchen"}),
        edges=frozenset(),
    )
    extra_edge = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Kitchen"}),
        edges=frozenset(
            {("Bedroom", "Kitchen"), ("Bedroom", "Bath")}
        ),
    )
    rows = []
    for label, graph in (
        ("delete_node_type", deleted_node),
        ("add_spurious_node_type", extra_node),
        ("relabel_one_node_type", relabeled_node),
    ):
        attacked = node_precision_recall_f1(graph, target)
        key = "recall" if label == "delete_node_type" else "precision"
        rows.append(
            _attack_row(
                "node_f1",
                label,
                key,
                baseline_nodes[key],
                attacked[key],
            )
        )
        rows.append(
            _attack_row(
                "node_f1",
                f"{label}_f1",
                "f1",
                baseline_nodes["f1"],
                attacked["f1"],
            )
        )
    for label, graph in (
        ("delete_adjacency_edge", deleted_edge),
        ("insert_spurious_adjacency", extra_edge),
    ):
        attacked = edge_precision_recall_f1(graph, target)
        key = "recall" if label == "delete_adjacency_edge" else "precision"
        rows.append(
            _attack_row(
                "edge_overlap",
                label,
                key,
                baseline_edges[key],
                attacked[key],
            )
        )
        rows.append(
            _attack_row(
                "edge_overlap",
                f"{label}_f1",
                "f1",
                baseline_edges["f1"],
                attacked["f1"],
            )
        )
    taxonomy, before, region, changed = _edge_outside_context()
    shifted = before.copy()
    shifted[8:56, 35:50] = 0
    shifted[8:56, 39:54] = taxonomy.class_id("room:Bedroom")
    outside = outside_layer_metrics(
        before,
        shifted,
        changed,
        region,
        before > 0,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    rows.append(
        _attack_row(
            "edge_overlap",
            "shift_edge_beyond_2px",
            "outside_edge_exact",
            1.0,
            outside["outside_edge_exact"],
        )
    )
    ged = housemind_style_ged(relabeled_node, target)
    rows.append(
        {
            "metric": "ged",
            "attack": "relabel_one_node_type",
            "submetric": "ged",
            "baseline": 0.0,
            "attacked": float(ged["ged"]),
            "delta": float(ged["ged"]),
            "direction": "increase",
            "sensitive": float(ged["ged"]) > 0.0,
            "false_pass": float(ged["ged"]) <= 0.0,
            "false_fail": False,
        }
    )
    return rows


__all__ = [
    "graph_attacks",
    "outside_attacks",
    "parse_args",
    "validity_attacks",
    "watertight_attacks",
]

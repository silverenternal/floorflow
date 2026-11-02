from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.housemind_structure import (
    _EXACT_NODE_PRODUCT_LIMIT,
    _LOCAL_SEARCH_STEPS,
    RoomTypeGraph,
    edge_set_overlap,
    housemind_style_ged,
    housemind_style_metrics_from_plan,
    node_set_f1,
    room_type_graph_from_plan,
)
from floorflow.geometry.wall_graph_readout import RasterWallPlan


def test_house_mind_type_graph_metrics_are_sets_not_instances() -> None:
    predicted = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Kitchen"}),
        edges=frozenset({("Bedroom", "Kitchen")}),
    )
    target = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Kitchen"}),
        edges=frozenset({("Bedroom", "Kitchen")}),
    )
    assert node_set_f1(predicted, target)["f1"] == 1.0
    assert edge_set_overlap(predicted, target)["overlap"] == 1.0
    assert housemind_style_ged(predicted, target)["ged"] == 0.0


def test_house_mind_type_graph_reports_missing_type_and_edge() -> None:
    predicted = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Kitchen"}),
        edges=frozenset(),
    )
    target = RoomTypeGraph(
        labels=frozenset({"Bedroom", "LivingRoom"}),
        edges=frozenset({("Bedroom", "LivingRoom")}),
    )
    nodes = node_set_f1(predicted, target)
    assert nodes["precision"] == 0.5
    assert nodes["recall"] == 0.5
    assert edge_set_overlap(predicted, target)["overlap"] == 0.0
    ged = housemind_style_ged(predicted, target)
    assert ged["ged"] == 2.0
    assert ged["node_substitution"] == 1
    assert ged["edge_insertion"] == 1
    assert ged["edge_deletion"] == 0


def test_house_mind_large_ged_is_deterministic_without_wall_clock() -> None:
    predicted = RoomTypeGraph(
        labels=frozenset({f"predicted-{index}" for index in range(7)}),
        edges=frozenset(
            {
                ("predicted-0", "predicted-1"),
                ("predicted-1", "predicted-2"),
                ("predicted-3", "predicted-4"),
                ("predicted-4", "predicted-5"),
            }
        ),
    )
    target = RoomTypeGraph(
        labels=frozenset({f"target-{index}" for index in range(7)}),
        edges=frozenset(
            {
                ("target-0", "target-1"),
                ("target-1", "target-2"),
                ("target-2", "target-3"),
                ("target-4", "target-5"),
            }
        ),
    )
    first = housemind_style_ged(predicted, target, timeout_seconds=0.0)
    second = housemind_style_ged(predicted, target, timeout_seconds=0.0)
    assert first == second
    assert first["exact"] is False
    assert first["fallback_reason"] == "deterministic_bounded_matching"


def test_house_mind_ged_contract_matches_runtime_budget() -> None:
    contract = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/housemind_style_ged_deterministic_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["exact_search"]["node_product_limit"] == (
        _EXACT_NODE_PRODUCT_LIMIT
    )
    assert contract["bounded_search"]["local_search_steps"] == _LOCAL_SEARCH_STEPS
    assert contract["exact_search"]["wall_clock_timeout"] is None


def test_room_type_graph_collapses_instances_of_the_same_type() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = np.zeros((32, 32), dtype=np.uint8)
    wall = taxonomy.class_id("wall")
    bedroom = taxonomy.class_id("room:Bedroom")
    labels[4:28, 4:14] = bedroom
    labels[4:28, 18:28] = bedroom
    labels[4:28, 4:6] = wall
    labels[4:28, 14:16] = wall
    labels[4:28, 18:20] = wall
    labels[4:28, 26:28] = wall
    labels[4:6, 4:28] = wall
    labels[26:28, 4:28] = wall
    from floorflow.evaluation.housemind_structure import room_type_graph

    graph = room_type_graph(labels, taxonomy, labels > 0)
    assert graph.labels == frozenset({"room:Bedroom"})
    assert graph.edges == frozenset()


def test_room_type_graph_can_come_from_explicit_plan() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    plan = RasterWallPlan(
        frame=32.0,
        walls=[],
        rooms=[
            {
                "label": "room:Bedroom",
                "polygon": [[0, 0], [16, 0], [16, 32], [0, 32]],
                "holes": [],
            },
            {
                "label": "room:Kitchen",
                "polygon": [[16, 0], [32, 0], [32, 32], [16, 32]],
                "holes": [],
            },
        ],
        openings=[],
    )
    graph = room_type_graph_from_plan(plan, taxonomy)
    assert graph.labels == frozenset({"room:Bedroom", "room:Kitchen"})
    assert graph.edges == frozenset(
        {("room:Bedroom", "room:Kitchen")}
    )
    target = np.zeros((32, 32), dtype=np.int64)
    target[:, :16] = taxonomy.room_ids[0]
    target[:, 16:] = taxonomy.room_ids[1]
    metrics = housemind_style_metrics_from_plan(
        plan,
        target,
        np.ones((32, 32), dtype=bool),
        taxonomy,
    )
    assert metrics["node_f1"] == 1.0
    assert metrics["edge_overlap"] == 1.0

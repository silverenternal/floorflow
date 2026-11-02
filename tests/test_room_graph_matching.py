from __future__ import annotations

from collections import Counter

import numpy as np
from shapely.geometry import box

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.room_graph_matching import (
    INSTANCE_PRIMARY_PROTOCOL,
    RoomGraph,
    RoomGraphNode,
    build_direct_room_graph,
    build_room_graph,
    exact_room_graph_edit_distance,
    match_room_graphs,
)
from floorflow.evaluation.room_graph_strata import (
    EditRegion,
    ambiguity_summary,
    ged_error_rows,
)
from floorflow.evaluation.sota_protocol import GRAPH_PROTOCOL


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


def test_instance_primary_protocol_matches_frozen_sota_contract() -> None:
    assert INSTANCE_PRIMARY_PROTOCOL == GRAPH_PROTOCOL


def _floorplan(taxonomy, *, swapped: bool = False):
    labels = np.zeros((64, 64), dtype=np.uint8)
    wall = taxonomy.class_id("wall")
    left_id = taxonomy.class_id("room:Kitchen" if swapped else "room:Bedroom")
    right_id = taxonomy.class_id("room:Bedroom" if swapped else "room:Kitchen")
    labels[8:56, 8:31] = left_id
    labels[8:56, 33:56] = right_id
    labels[8:56, 8:11] = wall
    labels[8:56, 31:33] = wall
    labels[8:56, 53:56] = wall
    labels[8:11, 8:56] = wall
    labels[53:56, 8:56] = wall
    return labels


def _manual_graph(labels: tuple[str, str], edges: set[tuple[int, int]]) -> RoomGraph:
    polygons = (box(0, 0, 10, 10), box(20, 0, 30, 10))
    nodes = tuple(
        RoomGraphNode(
            index=index,
            label=label,
            polygon=polygon,
            centroid=(polygon.centroid.x, polygon.centroid.y),
            label_purity=1.0,
            second_label_fraction=0.0,
        )
        for index, (label, polygon) in enumerate(zip(labels, polygons, strict=True))
    )
    return RoomGraph(nodes=nodes, edges=frozenset(edges), opening_count=0)


def test_identical_room_graphs_are_perfect_and_zero_edit() -> None:
    taxonomy = _taxonomy()
    labels = _floorplan(taxonomy)
    graph = build_room_graph(labels, taxonomy)

    for protocol in (
        "label_iou",
        "geometry_hungarian",
        "type_geometry_hungarian",
    ):
        metrics = match_room_graphs(graph, graph, protocol=protocol)
        assert metrics["node_f1"] == 1.0
        assert metrics["edge_overlap"] == 1.0
        assert metrics["node_substitutions"] == 0

    ged = exact_room_graph_edit_distance(graph, graph)
    assert ged["ged"] == 0.0
    assert ged["ged_exact"]
    assert ged["total"] == 0.0


def test_footprint_tiling_recovers_direct_room_adjacency() -> None:
    taxonomy = _taxonomy()
    labels = _floorplan(taxonomy)
    graph = build_direct_room_graph(
        labels,
        taxonomy,
        footprint=labels > 0,
        tile_footprint=True,
    )
    assert len(graph.nodes) == 2
    assert graph.edges == {(0, 1)}


def test_geometry_and_semantic_matching_expose_label_substitutions() -> None:
    prediction = _manual_graph(("room:Bedroom", "room:Kitchen"), {(0, 1)})
    target = _manual_graph(("room:Bedroom", "room:Bedroom"), {(0, 1)})

    semantic = match_room_graphs(
        prediction,
        target,
        protocol="label_iou",
    )
    geometry = match_room_graphs(
        prediction,
        target,
        protocol="geometry_hungarian",
    )
    assert semantic["node_f1"] == 0.5
    assert geometry["node_f1"] == 1.0
    assert geometry["node_substitutions"] == 1

    ged = exact_room_graph_edit_distance(prediction, target)
    assert ged["ged_exact"]
    assert ged["node_substitution"] == 1
    assert ged["ged"] == ged["total"]


def test_edit_region_bands_are_mutually_exclusive() -> None:
    region = EditRegion.from_normalized(
        [0.25, 0.25, 0.5, 0.5],
        width=100,
        height=100,
    )
    assert region.classify((50.0, 50.0)) == "inside"
    assert region.classify((24.0, 50.0)) == "boundary_1"
    assert region.classify((23.0, 50.0)) == "boundary_2"
    assert region.classify((20.0, 50.0)) == "boundary_5"
    assert region.classify((10.0, 50.0)) == "far_outside"


def test_ambiguity_and_ged_errors_use_indexed_graph_nodes() -> None:
    prediction = _manual_graph(("room:Bedroom", "room:Kitchen"), {(0, 1)})
    target = _manual_graph(("room:Bedroom", "room:Bedroom"), {(0, 1)})
    ambiguity = ambiguity_summary(prediction, target)
    assert ambiguity["target_same_type_adjacent_edges"] == 1

    ged = exact_room_graph_edit_distance(prediction, target)
    region = EditRegion(0.0, 0.0, 64.0, 64.0)
    rows = ged_error_rows(
        prediction,
        target,
        ged["mapping"],
        region,
        node_deletions=ged["_ged_node_deletions"],
        node_insertions=ged["_ged_node_insertions"],
        edge_deletions=ged["_ged_edge_deletions"],
        edge_insertions=ged["_ged_edge_insertions"],
    )
    substitutions = [row for row in rows if row["kind"] == "node_substitution"]
    assert len(substitutions) == 1
    assert all(row["location"] == "inside" for row in substitutions)


def test_edge_error_indices_stay_in_their_original_graph_domains() -> None:
    prediction = _manual_graph(("room:Bedroom", "room:Kitchen"), {(0, 1)})
    target = _manual_graph(("room:Bedroom", "room:Kitchen"), set())
    metrics = match_room_graphs(
        prediction,
        target,
        protocol="label_iou",
    )
    assert metrics["edge_false_positive"] == [(0, 1)]
    assert metrics["edge_false_negative"] == []


def test_edge_overlap_is_conditional_on_matched_target_nodes() -> None:
    taxonomy = _taxonomy()
    labels = _floorplan(taxonomy)
    prediction = build_direct_room_graph(
        labels,
        taxonomy,
        footprint=labels > 0,
        tile_footprint=True,
    )
    target = RoomGraph(
        nodes=prediction.nodes
        + (
            RoomGraphNode(
                index=2,
                label="room:Kitchen",
                polygon=box(40, 0, 50, 10),
                centroid=(45.0, 5.0),
                label_purity=1.0,
                second_label_fraction=0.0,
            ),
        ),
        edges=frozenset({(0, 1), (1, 2)}),
        opening_count=0,
    )
    metrics = match_room_graphs(
        prediction,
        target,
        protocol="geometry_hungarian",
    )
    assert metrics["matched_nodes"] == 2
    assert metrics["node_f1"] == 0.8
    assert metrics["edge_overlap"] == 1.0
    assert metrics["edge_false_negative"] == []


def test_ged_fallback_edge_indices_stay_in_their_original_domains() -> None:
    prediction = _manual_graph(("room:Bedroom", "room:Kitchen"), {(0, 1)})
    target = _manual_graph(("room:Bedroom", "room:Kitchen"), set())
    ged = exact_room_graph_edit_distance(prediction, target, max_nodes=0)
    assert not ged["ged_exact"]
    assert ged["_ged_edge_deletions"] == [(0, 1)]
    assert ged["_ged_edge_insertions"] == []

"""Deterministic room-graph refinement candidates for fixed-protocol audits."""

from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_graph_matching import (
    RoomGraph,
    build_direct_room_graph,
    build_room_graph,
    filter_room_graph,
)


def _prune_short_edges(graph: RoomGraph, minimum_contact: float) -> RoomGraph:
    if minimum_contact <= 0.0:
        return graph
    edges = set()
    for left, right in graph.edges:
        left_polygon = graph.nodes[left].polygon
        right_polygon = graph.nodes[right].polygon
        contact = left_polygon.boundary.intersection(
            right_polygon.buffer(1.0)
        ).length
        contact += right_polygon.boundary.intersection(
            left_polygon.buffer(1.0)
        ).length
        if 0.5 * contact >= minimum_contact:
            edges.add((left, right))
    return RoomGraph(
        nodes=graph.nodes,
        edges=frozenset(edges),
        opening_count=graph.opening_count,
    )


def _hybrid_wall_edges(
    refined: RoomGraph,
    wall_graph: RoomGraph,
    *,
    union_direct: bool,
) -> RoomGraph:
    wall_to_refined = {}
    for wall_node in wall_graph.nodes:
        best_index = None
        best_overlap = -1.0
        for refined_node in refined.nodes:
            overlap = wall_node.polygon.intersection(
                refined_node.polygon
            ).area
            if overlap > best_overlap:
                best_index = refined_node.index
                best_overlap = overlap
        if best_index is not None:
            wall_to_refined[wall_node.index] = best_index
    edges = set(refined.edges) if union_direct else set()
    for left, right in wall_graph.edges:
        mapped_left = wall_to_refined.get(left)
        mapped_right = wall_to_refined.get(right)
        if mapped_left is None or mapped_right is None:
            continue
        if mapped_left != mapped_right:
            edges.add(tuple(sorted((mapped_left, mapped_right))))
    return RoomGraph(
        nodes=refined.nodes,
        edges=frozenset(edges),
        opening_count=refined.opening_count,
    )


def refine_room_graph(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    footprint: np.ndarray | None = None,
    method: str = "tiled_purity_050",
) -> RoomGraph:
    """Build a deterministic graph readout without changing raster outputs."""

    if method == "wall_wall_primary":
        return build_room_graph(labels, taxonomy)
    if method not in {
        "direct_tiled",
        "tiled_no_undefined",
        "tiled_purity_050",
        "tiled_no_undefined_purity_050",
        "tiled_purity_050_contact_02",
        "tiled_purity_050_contact_04",
        "tiled_purity_050_contact_08",
        "tiled_purity_050_wall_edges",
        "tiled_purity_050_wall_union",
    }:
        raise ValueError(f"unknown graph refinement method: {method}")
    graph = build_direct_room_graph(
        labels,
        taxonomy,
        adjacency_tolerance=2.0,
        footprint=footprint,
        tile_footprint=True,
    )
    keep = [node.index for node in graph.nodes]
    if "no_undefined" in method or method == "tiled_purity_050":
        keep = [
            index
            for index in keep
            if graph.nodes[index].label != "undefined"
        ]
    if "purity_050" in method:
        keep = [
            index
            for index in keep
            if graph.nodes[index].label_purity >= 0.5
        ]
    graph = filter_room_graph(graph, keep)
    if method.endswith(("wall_edges", "wall_union")):
        wall_graph = build_room_graph(labels, taxonomy)
        return _hybrid_wall_edges(
            graph,
            wall_graph,
            union_direct=method.endswith("wall_union"),
        )
    for suffix, threshold in (
        ("contact_02", 2.0),
        ("contact_04", 4.0),
        ("contact_08", 8.0),
    ):
        if method.endswith(suffix):
            return _prune_short_edges(graph, threshold)
    return graph


def graph_refinement_methods() -> tuple[str, ...]:
    return (
        "wall_wall_primary",
        "direct_tiled",
        "tiled_no_undefined",
        "tiled_purity_050",
        "tiled_no_undefined_purity_050",
        "tiled_purity_050_contact_02",
        "tiled_purity_050_contact_04",
        "tiled_purity_050_contact_08",
        "tiled_purity_050_wall_edges",
        "tiled_purity_050_wall_union",
    )


__all__ = ["graph_refinement_methods", "refine_room_graph"]

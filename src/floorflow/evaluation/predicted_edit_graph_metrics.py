"""Canonical type-graph and instance diagnostics for predicted edits."""

from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.housemind_structure import (
    housemind_style_metrics,
)
from floorflow.evaluation.room_graph_edit import (
    exact_room_graph_edit_distance,
)
from floorflow.evaluation.room_graph_matching import match_room_graphs
from floorflow.inference.room_graph_refinement import refine_room_graph


def predicted_edit_graph_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    visible: np.ndarray,
    footprint: np.ndarray,
    timeout_seconds: float,
) -> dict[str, dict]:
    """Return canonical type metrics plus baseline/refined instance diagnostics."""

    type_metrics = housemind_style_metrics(
        visible,
        target,
        footprint,
        taxonomy,
        ged_timeout_seconds=timeout_seconds,
    )
    baseline = refine_room_graph(
        predicted,
        taxonomy,
        method="wall_wall_primary",
    )
    refined = refine_room_graph(
        predicted,
        taxonomy,
        footprint=predicted > 0,
        method="direct_tiled",
    )
    target_graph = refine_room_graph(
        target,
        taxonomy,
        footprint=footprint,
        method="direct_tiled",
    )
    output = {}
    for name, instance in (
        ("baseline", baseline),
        ("refined", refined),
    ):
        matched = match_room_graphs(
            instance,
            target_graph,
            protocol="type_geometry_hungarian",
        )
        ged = exact_room_graph_edit_distance(
            instance,
            target_graph,
            max_nodes=32,
            timeout_seconds=timeout_seconds,
        )
        output[name] = {
            "node_f1": float(type_metrics["node_f1"]),
            "edge_overlap": float(type_metrics["edge_overlap"]),
            "ged": float(type_metrics["ged"]),
            "ged_exact": bool(type_metrics["exact"]),
            "ged_fallback_reason": type_metrics["fallback_reason"],
            "node_deletion": int(type_metrics["node_deletion"]),
            "node_insertion": int(type_metrics["node_insertion"]),
            "node_substitution": int(type_metrics["node_substitution"]),
            "edge_deletion": int(type_metrics["edge_deletion"]),
            "edge_insertion": int(type_metrics["edge_insertion"]),
            "predicted_edges": int(type_metrics["edge_predicted_edges"]),
            "target_edges": int(type_metrics["edge_target_edges"]),
            "predicted_nodes": int(type_metrics["node_predicted_nodes"]),
            "target_nodes": int(type_metrics["node_target_nodes"]),
            "matched_nodes": int(type_metrics["node_intersection"]),
            "instance_node_f1": float(matched["node_f1"]),
            "instance_edge_overlap": float(matched["edge_overlap"]),
            "instance_ged": float(ged["ged"]),
            "instance_matched_nodes": int(matched["matched_nodes"]),
            "instance_predicted_nodes": len(instance.nodes),
            "instance_target_nodes": len(target_graph.nodes),
        }
    return output


__all__ = ["predicted_edit_graph_metrics"]

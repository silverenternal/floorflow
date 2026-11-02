"""Exact and fallback room-graph edit distance."""

from __future__ import annotations

import time

import networkx as nx

from floorflow.evaluation.room_graph_matching import (
    RoomGraph,
    _decomposition,
    _fallback_ged,
    _path_decomposition,
)


def exact_room_graph_edit_distance(
    predicted: RoomGraph,
    target: RoomGraph,
    *,
    max_nodes: int = 24,
    timeout_seconds: float | None = 15.0,
) -> dict:
    """Compute unit-cost attributed GED with an explicit fallback."""

    total_nodes = len(predicted.nodes) + len(target.nodes)
    if total_nodes > max_nodes:
        return _fallback_ged(
            predicted,
            target,
            reason=f"node_budget_exceeded:{total_nodes}>{max_nodes}",
        )
    predicted_graph = nx.Graph()
    target_graph = nx.Graph()
    for node in predicted.nodes:
        predicted_graph.add_node(node.index, room_type=node.label)
    for node in target.nodes:
        target_graph.add_node(node.index, room_type=node.label)
    predicted_graph.add_edges_from(predicted.edges)
    target_graph.add_edges_from(target.edges)
    if not predicted_graph and not target_graph:
        return {
            "ged": 0.0,
            "ged_exact": True,
            "ged_fallback_reason": None,
            "mapping": {},
            "_ged_node_deletions": [],
            "_ged_node_insertions": [],
            "_ged_edge_deletions": [],
            "_ged_edge_insertions": [],
            **_decomposition(predicted, target, {}),
        }
    try:
        started = time.perf_counter()
        paths = nx.optimize_edit_paths(
            predicted_graph,
            target_graph,
            node_subst_cost=lambda left, right: (
                0 if left["room_type"] == right["room_type"] else 1
            ),
            node_del_cost=lambda _node: 1,
            node_ins_cost=lambda _node: 1,
            edge_del_cost=lambda _edge: 1,
            edge_ins_cost=lambda _edge: 1,
            edge_subst_cost=lambda _left, _right: 0,
            timeout=timeout_seconds,
        )
        best = None
        for node_pairs, edge_pairs, cost in paths:
            if cost is None:
                continue
            if best is None or cost <= best[2]:
                best = (node_pairs, edge_pairs, cost)
        if best is None:
            raise RuntimeError("networkx returned no edit path")
        decomposition = _path_decomposition(
            predicted,
            target,
            best[0],
            best[1],
        )
        elapsed = time.perf_counter() - started
        exact = timeout_seconds is None or elapsed < float(timeout_seconds)
        return {
            "ged": float(best[2]),
            "ged_exact": exact,
            "ged_fallback_reason": (
                None if exact else "exact_timeout_returned_best_bound"
            ),
            **decomposition,
        }
    except (nx.NetworkXError, RuntimeError, ValueError) as error:
        return _fallback_ged(
            predicted,
            target,
            reason=f"exact_failed:{type(error).__name__}",
        )


__all__ = ["exact_room_graph_edit_distance"]

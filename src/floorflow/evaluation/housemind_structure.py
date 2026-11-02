"""HouseMind-style room-type graph metrics and a canonical companion view."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_graph_matching import build_direct_room_graph


@dataclass(frozen=True)
class RoomTypeGraph:
    labels: frozenset[str]
    edges: frozenset[tuple[str, str]]


_EXACT_NODE_PRODUCT_LIMIT = 36
_LOCAL_SEARCH_STEPS = 256


def room_type_graph(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    footprint: np.ndarray,
    *,
    adjacency_tolerance: float = 2.0,
) -> RoomTypeGraph:
    """Collapse the canonical direct-tiled room graph to unique room types."""

    graph = build_direct_room_graph(
        labels,
        taxonomy,
        adjacency_tolerance=adjacency_tolerance,
        footprint=footprint,
        tile_footprint=True,
    )
    room_labels = frozenset(node.label for node in graph.nodes)
    type_edges = frozenset(
        tuple(sorted((graph.nodes[left].label, graph.nodes[right].label)))
        for left, right in graph.edges
        if graph.nodes[left].label != graph.nodes[right].label
    )
    return RoomTypeGraph(labels=room_labels, edges=type_edges)


def room_type_graph_from_plan(
    plan,
    taxonomy: FloorplanTaxonomy,
    *,
    adjacency_tolerance: float = 2.0,
) -> RoomTypeGraph:
    """Build the same room-type graph from explicit plan polygons."""

    labels = set()
    polygons = []
    for room in plan.rooms:
        label = str(room["label"])
        if not taxonomy.is_room(label):
            continue
        polygon = Polygon(room["polygon"], room.get("holes", ()))
        if polygon.is_empty or not polygon.is_valid:
            continue
        labels.add(label)
        polygons.append((label, polygon))
    edges = set()
    for index, (left_label, left) in enumerate(polygons):
        for right_label, right in polygons[index + 1 :]:
            if left_label == right_label:
                continue
            if left.distance(right) <= float(adjacency_tolerance):
                edges.add(tuple(sorted((left_label, right_label))))
    return RoomTypeGraph(
        labels=frozenset(labels),
        edges=frozenset(edges),
    )


def node_set_f1(
    predicted: RoomTypeGraph,
    target: RoomTypeGraph,
) -> dict[str, float | int]:
    """HouseMind Appendix B.2 Node F1 over unique room types."""

    if not predicted.labels and not target.labels:
        return {
            "intersection": 0,
            "predicted_nodes": 0,
            "target_nodes": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    intersection = predicted.labels & target.labels
    precision = len(intersection) / len(predicted.labels) if predicted.labels else 0.0
    recall = len(intersection) / len(target.labels) if target.labels else 0.0
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return {
        "intersection": len(intersection),
        "predicted_nodes": len(predicted.labels),
        "target_nodes": len(target.labels),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def edge_set_overlap(
    predicted: RoomTypeGraph,
    target: RoomTypeGraph,
) -> dict[str, float | int]:
    """Edge precision, recall, F1, and Jaccard over adjacency pairs."""

    intersection = len(predicted.edges & target.edges)
    precision = (
        intersection / len(predicted.edges) if predicted.edges else 0.0
    )
    recall = intersection / len(target.edges) if target.edges else 0.0
    denominator = precision + recall
    f1 = (
        2.0 * precision * recall / denominator
        if denominator
        else (1.0 if not predicted.edges and not target.edges else 0.0)
    )
    union = predicted.edges | target.edges
    return {
        "intersection": intersection,
        "predicted_edges": len(predicted.edges),
        "target_edges": len(target.edges),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "overlap": (
            float(intersection / len(union))
            if union
            else 1.0
        ),
    }


def _networkx_graph(graph: RoomTypeGraph) -> nx.Graph:
    result = nx.Graph()
    for label in sorted(graph.labels):
        result.add_node(label, room_type=label)
    result.add_edges_from(sorted(graph.edges))
    return result


def _mapping_decomposition(
    predicted: RoomTypeGraph,
    target: RoomTypeGraph,
    mapping: dict[str, str],
) -> dict[str, float | int]:
    predicted_edges = {
        tuple(sorted((left, right)))
        for left, right in predicted.edges
    }
    target_edges = {
        tuple(sorted((left, right)))
        for left, right in target.edges
    }
    mapped_edges = {
        tuple(sorted((mapping[left], mapping[right])))
        for left, right in predicted_edges
        if left in mapping and right in mapping
    }
    node_deletion = len(predicted.labels) - len(mapping)
    node_insertion = len(target.labels) - len(mapping)
    node_substitution = sum(
        left != right for left, right in mapping.items()
    )
    edge_deletion = len(mapped_edges - target_edges)
    edge_insertion = len(target_edges - mapped_edges)
    total = (
        node_deletion
        + node_insertion
        + node_substitution
        + edge_deletion
        + edge_insertion
    )
    return {
        "ged": float(total),
        "node_substitution": int(node_substitution),
        "node_deletion": int(node_deletion),
        "node_insertion": int(node_insertion),
        "edge_deletion": int(edge_deletion),
        "edge_insertion": int(edge_insertion),
    }


def _exact_label_mapping(
    predicted: RoomTypeGraph,
    target: RoomTypeGraph,
) -> dict[str, str]:
    return {
        label: label
        for label in sorted(predicted.labels & target.labels)
    }


def _hungarian_mapping(
    predicted: RoomTypeGraph,
    target: RoomTypeGraph,
    *,
    degree_tiebreak: bool,
) -> dict[str, str]:
    predicted_labels = sorted(predicted.labels)
    target_labels = sorted(target.labels)
    if not predicted_labels or not target_labels:
        return {}
    costs = np.ones(
        (len(predicted_labels), len(target_labels)),
        dtype=np.float64,
    )
    predicted_degree = _degree_map(predicted)
    target_degree = _degree_map(target)
    max_degree = max(
        1,
        max(predicted_degree.values(), default=0),
        max(target_degree.values(), default=0),
    )
    for row, left in enumerate(predicted_labels):
        for column, right in enumerate(target_labels):
            if left == right:
                costs[row, column] = 0.0
            if degree_tiebreak:
                costs[row, column] += 1e-3 * abs(
                    predicted_degree[left] - target_degree[right]
                ) / max_degree
    rows, columns = linear_sum_assignment(costs)
    return {
        predicted_labels[int(row)]: target_labels[int(column)]
        for row, column in zip(rows, columns, strict=True)
    }


def _degree_map(graph: RoomTypeGraph) -> dict[str, int]:
    degree = {label: 0 for label in graph.labels}
    for left, right in graph.edges:
        degree[left] += 1
        degree[right] += 1
    return degree


def _candidate_mappings(
    predicted: RoomTypeGraph,
    target: RoomTypeGraph,
) -> list[dict[str, str]]:
    return [
        _exact_label_mapping(predicted, target),
        _hungarian_mapping(predicted, target, degree_tiebreak=False),
        _hungarian_mapping(predicted, target, degree_tiebreak=True),
    ]


def _mapping_key(mapping: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(mapping.items()))


def _local_search_mapping(
    predicted: RoomTypeGraph,
    target: RoomTypeGraph,
    seed: dict[str, str],
) -> dict[str, str]:
    mapping = dict(seed)
    best = _mapping_decomposition(predicted, target, mapping)
    for _ in range(_LOCAL_SEARCH_STEPS):
        predicted_labels = sorted(predicted.labels)
        target_labels = sorted(target.labels)
        unmatched_predicted = [
            label for label in predicted_labels if label not in mapping
        ]
        unmatched_target = [
            label for label in target_labels if label not in mapping.values()
        ]
        choices: list[tuple[tuple[str, str | None, str | None], dict[str, str]]] = []
        for left in sorted(mapping):
            choices.append(((left, mapping[left], None), {left: None}))
            for new_right in unmatched_target:
                choices.append(((left, mapping[left], new_right), {left: new_right}))
        for left in unmatched_predicted:
            for right in unmatched_target:
                choices.append(((left, None, right), {left: right}))
        for left in sorted(mapping):
            for other_left in sorted(mapping):
                if left < other_left:
                    choices.append(
                        (
                            (left, mapping[left], mapping[other_left]),
                            {
                                left: mapping[other_left],
                                other_left: mapping[left],
                            },
                        )
                    )
        choices.sort(
            key=lambda item: (
                item[0][0],
                item[0][1] or "",
                item[0][2] or "",
            )
        )
        improved = False
        for _key, update in choices:
            candidate = dict(mapping)
            for left, right in update.items():
                candidate.pop(left, None)
                if right is not None:
                    occupant = next(
                        (
                            other
                            for other, current in candidate.items()
                            if current == right
                        ),
                        None,
                    )
                    if occupant is not None:
                        candidate.pop(occupant, None)
                    candidate[left] = right
            decomposition = _mapping_decomposition(
                predicted,
                target,
                candidate,
            )
            if (
                float(decomposition["ged"]) < float(best["ged"])
                or (
                    float(decomposition["ged"]) == float(best["ged"])
                    and _mapping_key(candidate) < _mapping_key(mapping)
                )
            ):
                mapping = candidate
                best = decomposition
                improved = True
                break
        if not improved:
            break
    return mapping


def _deterministic_bounded_ged(
    predicted: RoomTypeGraph,
    target: RoomTypeGraph,
) -> dict[str, float | bool | int | str]:
    candidates = [
        _local_search_mapping(predicted, target, seed)
        for seed in _candidate_mappings(predicted, target)
    ]
    best = min(
        candidates,
        key=lambda mapping: (
            float(_mapping_decomposition(predicted, target, mapping)["ged"]),
            _mapping_key(mapping),
        ),
    )
    return {
        **_mapping_decomposition(predicted, target, best),
        "exact": False,
        "fallback_reason": "deterministic_bounded_matching",
    }


def housemind_style_ged(
    predicted: RoomTypeGraph,
    target: RoomTypeGraph,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, float | bool | int | str]:
    """Compute GED with a deterministic exact/fallback budget.

    Legacy callers may still pass ``timeout_seconds``. It is retained for
    API compatibility but intentionally does not control the result because
    wall-clock pruning makes the same graph pair non-reproducible.
    """

    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    node_product = len(predicted.labels) * len(target.labels)
    if node_product > _EXACT_NODE_PRODUCT_LIMIT:
        return _deterministic_bounded_ged(predicted, target)

    predicted_graph = _networkx_graph(predicted)
    target_graph = _networkx_graph(target)
    try:
        paths = nx.optimize_edit_paths(
            predicted_graph,
            target_graph,
            node_subst_cost=lambda left, right: (
                0 if left["room_type"] == right["room_type"] else 1
            ),
            node_del_cost=lambda _node: 1,
            node_ins_cost=lambda _node: 1,
            timeout=None,
        )
        best = None
        for node_pairs, edge_pairs, cost in paths:
            if cost is None:
                continue
            if best is None or cost <= best[2]:
                best = (node_pairs, edge_pairs, cost)
        if best is None:
            raise RuntimeError("NetworkX returned no edit path")
        mapping = {
            left: right
            for left, right in best[0]
            if left is not None and right is not None
        }
        edge_deletion = sum(
            left is not None and right is None
            for left, right in best[1]
        )
        edge_insertion = sum(
            left is None and right is not None
            for left, right in best[1]
        )
        return {
            "ged": float(best[2]),
            "exact": True,
            "fallback_reason": None,
            "node_substitution": sum(
                predicted_graph.nodes[left]["room_type"]
                != target_graph.nodes[right]["room_type"]
                for left, right in mapping.items()
            ),
            "node_deletion": sum(
                left is not None and right is None
                for left, right in best[0]
            ),
            "node_insertion": sum(
                left is None and right is not None
                for left, right in best[0]
            ),
            "edge_deletion": edge_deletion,
            "edge_insertion": edge_insertion,
        }
    except (nx.NetworkXError, RuntimeError, ValueError):
        return _deterministic_bounded_ged(predicted, target)


def housemind_style_metrics(
    predicted_labels: np.ndarray,
    target_labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    ged_timeout_seconds: float = 5.0,
) -> dict[str, float | int | bool | None]:
    predicted = room_type_graph(predicted_labels, taxonomy, footprint)
    target = room_type_graph(target_labels, taxonomy, footprint)
    nodes = node_set_f1(predicted, target)
    edges = edge_set_overlap(predicted, target)
    ged = housemind_style_ged(
        predicted,
        target,
        timeout_seconds=ged_timeout_seconds,
    )
    return {
        **{f"node_{key}": value for key, value in nodes.items()},
        **{f"edge_{key}": value for key, value in edges.items()},
        **ged,
    }


def housemind_style_metrics_from_plan(
    predicted_plan,
    target_labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    ged_timeout_seconds: float = 5.0,
) -> dict[str, float | int | bool | None]:
    predicted = room_type_graph_from_plan(predicted_plan, taxonomy)
    target = room_type_graph(target_labels, taxonomy, footprint)
    nodes = node_set_f1(predicted, target)
    edges = edge_set_overlap(predicted, target)
    ged = housemind_style_ged(
        predicted,
        target,
        timeout_seconds=ged_timeout_seconds,
    )
    return {
        **{f"node_{key}": value for key, value in nodes.items()},
        **{f"edge_{key}": value for key, value in edges.items()},
        **ged,
    }


__all__ = [
    "RoomTypeGraph",
    "edge_set_overlap",
    "housemind_style_ged",
    "housemind_style_metrics",
    "housemind_style_metrics_from_plan",
    "node_set_f1",
    "room_type_graph",
    "room_type_graph_from_plan",
]

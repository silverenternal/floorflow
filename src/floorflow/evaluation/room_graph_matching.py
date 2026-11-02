"""Room-graph construction and matching-protocol diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.room_polygons import room_polygon_rows
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan

MATCHING_PROTOCOLS = (
    "label_iou",
    "geometry_hungarian",
    "type_geometry_hungarian",
)
INSTANCE_PRIMARY_PROTOCOL = "type_geometry_hungarian"
IOU_THRESHOLD = 0.5
TYPE_BONUS = 0.1


@dataclass(frozen=True)
class RoomGraphNode:
    index: int
    label: str
    polygon: Polygon
    centroid: tuple[float, float]
    label_purity: float
    second_label_fraction: float


@dataclass(frozen=True)
class RoomGraph:
    nodes: tuple[RoomGraphNode, ...]
    edges: frozenset[tuple[int, int]]
    opening_count: int


def _label_statistics(
    polygon: Polygon,
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> tuple[float, float]:
    height, width = labels.shape
    x0, y0, x1, y1 = polygon.bounds
    left = max(0, int(np.floor(x0)) - 1)
    top = max(0, int(np.floor(y0)) - 1)
    right = min(width, int(np.ceil(x1)) + 2)
    bottom = min(height, int(np.ceil(y1)) + 2)
    local = np.zeros((max(1, bottom - top), max(1, right - left)), dtype=np.uint8)
    points = np.rint(
        np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
        - np.asarray([left, top], dtype=np.float64)
    ).astype(np.int32)
    cv2.fillPoly(local, [points], 1)
    values = labels[top:bottom, left:right][local.astype(bool)]
    room_values = values[np.isin(values, np.asarray(taxonomy.room_ids))]
    if room_values.size == 0:
        return 0.0, 0.0
    counts = np.bincount(room_values.reshape(-1), minlength=taxonomy.num_classes)
    ranked = np.sort(counts[np.asarray(taxonomy.room_ids)])[::-1]
    purity = float(ranked[0] / room_values.size)
    second = float(ranked[1] / room_values.size) if ranked.size > 1 else 0.0
    return purity, second


def build_graph_from_rows(
    rows: Iterable[dict],
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    adjacency_tolerance: float = 2.0,
    opening_count: int = 0,
) -> RoomGraph:
    nodes = []
    for room in rows:
        polygon = Polygon(room["polygon"])
        if polygon.is_empty or polygon.area <= 0.0:
            continue
        purity, second = _label_statistics(polygon, labels, taxonomy)
        nodes.append(
            RoomGraphNode(
                index=len(nodes),
                label=str(room["label"]),
                polygon=polygon,
                centroid=(float(polygon.centroid.x), float(polygon.centroid.y)),
                label_purity=purity,
                second_label_fraction=second,
            )
        )
    edges = set()
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            distance = nodes[left].polygon.boundary.distance(nodes[right].polygon.boundary)
            if distance <= float(adjacency_tolerance):
                edges.add((left, right))
    return RoomGraph(
        nodes=tuple(nodes),
        edges=frozenset(edges),
        opening_count=int(opening_count),
    )


def build_room_graph(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    adjacency_tolerance: float = 2.0,
) -> RoomGraph:
    """Convert raster labels to a wall-face room adjacency graph."""

    plan = raster_labels_to_wall_plan(labels, taxonomy)
    return build_graph_from_rows(
        plan.rooms,
        labels,
        taxonomy,
        adjacency_tolerance=adjacency_tolerance,
        opening_count=len(plan.openings),
    )


def build_direct_room_graph(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    adjacency_tolerance: float = 2.0,
    footprint: np.ndarray | None = None,
    tile_footprint: bool = False,
) -> RoomGraph:
    """Build room nodes directly from connected room-label polygons."""

    rows = room_polygon_rows(
        labels,
        taxonomy,
        footprint=footprint,
        tile_footprint=tile_footprint,
    )
    return build_graph_from_rows(
        rows,
        labels,
        taxonomy,
        adjacency_tolerance=adjacency_tolerance,
    )


def filter_room_graph(
    graph: RoomGraph,
    keep_indices: Iterable[int],
) -> RoomGraph:
    """Drop selected nodes and remap the retained adjacency graph."""

    kept = sorted({int(index) for index in keep_indices})
    remap = {old: new for new, old in enumerate(kept)}
    nodes = tuple(
        RoomGraphNode(
            index=remap[node.index],
            label=node.label,
            polygon=node.polygon,
            centroid=node.centroid,
            label_purity=node.label_purity,
            second_label_fraction=node.second_label_fraction,
        )
        for node in graph.nodes
        if node.index in remap
    )
    edges = {
        tuple(sorted((remap[left], remap[right])))
        for left, right in graph.edges
        if left in remap and right in remap
    }
    return RoomGraph(
        nodes=nodes,
        edges=frozenset(edges),
        opening_count=graph.opening_count,
    )


def room_iou_matrix(predicted: RoomGraph, target: RoomGraph) -> np.ndarray:
    matrix = np.zeros((len(predicted.nodes), len(target.nodes)), dtype=np.float64)
    for left, predicted_node in enumerate(predicted.nodes):
        for right, target_node in enumerate(target.nodes):
            union = predicted_node.polygon.union(target_node.polygon).area
            if union > 0.0:
                matrix[left, right] = (
                    predicted_node.polygon.intersection(target_node.polygon).area
                    / union
                )
    return matrix


def hungarian_mapping(
    predicted: RoomGraph,
    target: RoomGraph,
    *,
    protocol: str,
    iou: np.ndarray | None = None,
) -> dict[int, int]:
    if not predicted.nodes or not target.nodes:
        return {}
    iou = room_iou_matrix(predicted, target) if iou is None else np.asarray(iou)
    if protocol == "label_iou":
        similarity = np.zeros_like(iou)
        for left, predicted_node in enumerate(predicted.nodes):
            for right, target_node in enumerate(target.nodes):
                if predicted_node.label == target_node.label:
                    similarity[left, right] = iou[left, right]
    elif protocol == "geometry_hungarian":
        similarity = iou.copy()
    elif protocol == "type_geometry_hungarian":
        similarity = iou.copy()
        for left, predicted_node in enumerate(predicted.nodes):
            for right, target_node in enumerate(target.nodes):
                if predicted_node.label == target_node.label:
                    similarity[left, right] += TYPE_BONUS
    else:
        raise ValueError(f"unknown matching protocol: {protocol}")
    rows, columns = linear_sum_assignment(-similarity)
    mapping = {}
    for row, column in zip(rows, columns, strict=True):
        if iou[row, column] < IOU_THRESHOLD:
            continue
        if protocol == "label_iou" and (
            predicted.nodes[row].label != target.nodes[column].label
        ):
            continue
        mapping[int(row)] = int(column)
    return mapping


def mapped_edge_sets(
    predicted: RoomGraph,
    target: RoomGraph,
    mapping: dict[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    predicted_edges = {
        tuple(sorted((mapping[left], mapping[right])))
        for left, right in predicted.edges
        if left in mapping and right in mapping
    }
    mapped_targets = set(mapping.values())
    target_edges = {
        tuple(sorted((left, right)))
        for left, right in target.edges
        if left in mapped_targets and right in mapped_targets
    }
    return predicted_edges, target_edges


def _node_f1(
    predicted_count: int,
    target_count: int,
    matched_count: int,
) -> tuple[float, float, float]:
    precision = matched_count / predicted_count if predicted_count else float(not target_count)
    recall = matched_count / target_count if target_count else float(not predicted_count)
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return float(precision), float(recall), float(f1)


def match_room_graphs(
    predicted: RoomGraph,
    target: RoomGraph,
    *,
    protocol: str,
) -> dict:
    """Score a graph under one pre-registered node-matching protocol."""

    iou = room_iou_matrix(predicted, target)
    mapping = hungarian_mapping(
        predicted,
        target,
        protocol=protocol,
        iou=iou,
    )
    precision, recall, f1 = _node_f1(
        len(predicted.nodes),
        len(target.nodes),
        len(mapping),
    )
    mapped_edges, target_edges = mapped_edge_sets(predicted, target, mapping)
    predicted_edge_false_positive = sorted(
        (left, right)
        for left, right in predicted.edges
        if left in mapping
        and right in mapping
        and tuple(sorted((mapping[left], mapping[right]))) not in target_edges
    )
    target_edge_false_negative = sorted(target_edges - mapped_edges)
    edge_union = mapped_edges | target_edges
    edge_overlap = (
        len(mapped_edges & target_edges) / len(edge_union)
        if edge_union
        else 1.0
    )
    substitutions = sum(
        predicted.nodes[left].label != target.nodes[right].label
        for left, right in mapping.items()
    )
    return {
        "protocol": protocol,
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "edge_overlap": float(edge_overlap),
        "matched_nodes": len(mapping),
        "node_substitutions": int(substitutions),
        "predicted_edges": len(mapped_edges),
        "target_edges": len(target_edges),
        "mapping": mapping,
        "node_false_positive_indices": [
            index for index in range(len(predicted.nodes)) if index not in mapping
        ],
        "node_false_negative_indices": [
            index for index in range(len(target.nodes)) if index not in mapping.values()
        ],
        "edge_false_positive": predicted_edge_false_positive,
        "edge_false_negative": target_edge_false_negative,
    }


def _decomposition(
    predicted: RoomGraph,
    target: RoomGraph,
    mapping: dict[int, int],
) -> dict[str, int | float]:
    mapped_edges, target_edges = mapped_edge_sets(predicted, target, mapping)
    node_deletion = len(predicted.nodes) - len(mapping)
    node_insertion = len(target.nodes) - len(mapping)
    substitution = sum(
        predicted.nodes[left].label != target.nodes[right].label
        for left, right in mapping.items()
    )
    edge_deletion = len(mapped_edges - target_edges)
    edge_insertion = len(target_edges - mapped_edges)
    total = (
        node_deletion
        + node_insertion
        + substitution
        + edge_deletion
        + edge_insertion
    )
    return {
        "node_deletion": int(node_deletion),
        "node_insertion": int(node_insertion),
        "node_substitution": int(substitution),
        "edge_deletion": int(edge_deletion),
        "edge_insertion": int(edge_insertion),
        "total": float(total),
    }


def _path_decomposition(
    predicted: RoomGraph,
    target: RoomGraph,
    node_pairs: list[tuple[int | None, int | None]],
    edge_pairs: list[
        tuple[
            tuple[int, int] | None,
            tuple[int, int] | None,
        ]
    ],
) -> dict:
    mapping = {
        int(left): int(right)
        for left, right in node_pairs
        if left is not None and right is not None
    }
    node_deletions = [
        int(left)
        for left, right in node_pairs
        if left is not None and right is None
    ]
    node_insertions = [
        int(right)
        for left, right in node_pairs
        if left is None and right is not None
    ]
    substitutions = [
        (int(left), int(right))
        for left, right in node_pairs
        if left is not None
        and right is not None
        and predicted.nodes[left].label != target.nodes[right].label
    ]
    edge_deletions = [
        tuple(sorted((int(left), int(right))))
        for predicted_edge, target_edge in edge_pairs
        if predicted_edge is not None and target_edge is None
        for left, right in [predicted_edge]
    ]
    edge_insertions = [
        tuple(sorted((int(left), int(right))))
        for predicted_edge, target_edge in edge_pairs
        if predicted_edge is None and target_edge is not None
        for left, right in [target_edge]
    ]
    decomposition = {
        "node_deletion": len(node_deletions),
        "node_insertion": len(node_insertions),
        "node_substitution": len(substitutions),
        "edge_deletion": len(edge_deletions),
        "edge_insertion": len(edge_insertions),
    }
    decomposition["total"] = float(sum(decomposition.values()))
    return {
        **decomposition,
        "mapping": mapping,
        "_ged_node_deletions": node_deletions,
        "_ged_node_insertions": node_insertions,
        "_ged_edge_deletions": edge_deletions,
        "_ged_edge_insertions": edge_insertions,
    }


def _candidate_mappings(
    predicted: RoomGraph,
    target: RoomGraph,
) -> Iterable[dict[int, int]]:
    for protocol in MATCHING_PROTOCOLS:
        yield hungarian_mapping(predicted, target, protocol=protocol)


def _improve_mapping(
    predicted: RoomGraph,
    target: RoomGraph,
    seed: dict[int, int],
    *,
    max_steps: int = 256,
) -> dict[int, int]:
    """Deterministic one-swap fallback when exact GED exceeds the budget."""

    mapping = dict(seed)
    best_cost = float(_decomposition(predicted, target, mapping)["total"])
    for _ in range(max_steps):
        improved = False
        unmatched_predicted = [
            index for index in range(len(predicted.nodes)) if index not in mapping
        ]
        unmatched_target = [
            index for index in range(len(target.nodes)) if index not in mapping.values()
        ]
        choices: list[tuple[int, int | None, int | None]] = []
        for left, right in list(mapping.items()):
            choices.extend(
                (left, right, target_index)
                for target_index in unmatched_target
            )
            choices.append((left, right, None))
        choices.extend(
            (left, None, target_index)
            for left in unmatched_predicted
            for target_index in unmatched_target
        )
        for assigned_left in list(mapping):
            for assigned_right in list(mapping):
                if assigned_left < assigned_right:
                    choices.append((assigned_left, assigned_right, None))
        for choice in choices:
            candidate = dict(mapping)
            left, old_right, new_right = choice
            if old_right is None:
                candidate[left] = int(new_right)
            else:
                candidate.pop(left, None)
                if new_right is not None:
                    occupant = next(
                        (key for key, value in candidate.items() if value == new_right),
                        None,
                    )
                    if occupant is not None:
                        candidate.pop(occupant, None)
                    candidate[left] = int(new_right)
            cost = float(_decomposition(predicted, target, candidate)["total"])
            if cost < best_cost:
                mapping = candidate
                best_cost = cost
                improved = True
                break
        if not improved:
            break
    return mapping


def _fallback_ged(
    predicted: RoomGraph,
    target: RoomGraph,
    *,
    reason: str,
) -> dict:
    candidates = [
        _improve_mapping(predicted, target, mapping)
        for mapping in _candidate_mappings(predicted, target)
    ]
    best = min(
        candidates,
        key=lambda mapping: float(
            _decomposition(predicted, target, mapping)["total"]
        ),
        default={},
    )
    decomposition = _decomposition(predicted, target, best)
    mapped_edges, target_edges = mapped_edge_sets(predicted, target, best)
    mapped_targets = set(best.values())
    edge_deletions = sorted(
        (left, right)
        for left, right in predicted.edges
        if left in best
        and right in best
        and tuple(sorted((best[left], best[right]))) not in target_edges
    )
    edge_insertions = sorted(target_edges - mapped_edges)
    return {
        "ged": decomposition["total"],
        "ged_exact": False,
        "ged_fallback_reason": reason,
        "mapping": best,
        "_ged_node_deletions": [
            index for index in range(len(predicted.nodes)) if index not in best
        ],
        "_ged_node_insertions": [
            index for index in range(len(target.nodes)) if index not in mapped_targets
        ],
        "_ged_edge_deletions": edge_deletions,
        "_ged_edge_insertions": edge_insertions,
        **decomposition,
    }


def exact_room_graph_edit_distance(
    predicted: RoomGraph,
    target: RoomGraph,
    *,
    max_nodes: int = 24,
    timeout_seconds: float | None = 15.0,
) -> dict:
    """Compute unit-cost attributed GED through the dedicated edit module."""

    from floorflow.evaluation.room_graph_edit import (
        exact_room_graph_edit_distance as compute_ged,
    )

    return compute_ged(
        predicted,
        target,
        max_nodes=max_nodes,
        timeout_seconds=timeout_seconds,
    )


def protocol_matrix(
    predicted: RoomGraph,
    target: RoomGraph,
    *,
    ged_max_nodes: int = 24,
    ged_timeout_seconds: float | None = 15.0,
) -> dict:
    results = {
        protocol: match_room_graphs(predicted, target, protocol=protocol)
        for protocol in MATCHING_PROTOCOLS
    }
    ged = exact_room_graph_edit_distance(
        predicted,
        target,
        max_nodes=ged_max_nodes,
        timeout_seconds=ged_timeout_seconds,
    )
    return {"matching": results, "ged": ged}


def matching_distance_matrix(predicted: RoomGraph, target: RoomGraph) -> np.ndarray:
    """Expose the geometric affinity for downstream identity refinement."""

    return room_iou_matrix(predicted, target)


__all__ = [
    "INSTANCE_PRIMARY_PROTOCOL",
    "IOU_THRESHOLD",
    "MATCHING_PROTOCOLS",
    "RoomGraph",
    "RoomGraphNode",
    "build_direct_room_graph",
    "build_graph_from_rows",
    "build_room_graph",
    "exact_room_graph_edit_distance",
    "filter_room_graph",
    "match_room_graphs",
    "matching_distance_matrix",
    "protocol_matrix",
    "room_iou_matrix",
]

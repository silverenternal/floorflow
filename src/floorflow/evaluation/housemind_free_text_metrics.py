"""Image-level edit, locality, and room-type graph metrics for HouseMind data."""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation

from floorflow.inference.housemind_free_text_editor import (
    ROOM_COLORS,
    palette_labels,
)


def edit_metrics(
    before: np.ndarray,
    target_after: np.ndarray,
    predicted_after: np.ndarray,
) -> dict[str, float]:
    dark = np.maximum(before.max(axis=2), target_after.max(axis=2)) <= 30
    valid = ~dark
    expected = (np.any(before != target_after, axis=2)) & valid
    predicted = (np.any(before != predicted_after, axis=2)) & valid
    union = int((expected | predicted).sum())
    intersection = int((expected & predicted).sum())
    delta_iou = 1.0 if union == 0 else intersection / union
    delta_mse = float(
        np.square(expected.astype(np.float32) - predicted.astype(np.float32))[
            valid
        ].mean()
    )
    outside = valid & ~binary_dilation(expected, iterations=2)
    unchanged = np.all(predicted_after == before, axis=2)
    return {
        "delta_iou": float(delta_iou),
        "delta_mse": delta_mse,
        "outside_exact": float(unchanged[outside].mean()) if outside.any() else 1.0,
        "changed_pixels": float(predicted.sum()),
        "target_changed_pixels": float(expected.sum()),
    }


def _type_graph(image: np.ndarray) -> nx.Graph:
    labels, _ = palette_labels(image)
    graph = nx.Graph()
    types = [
        name for index, name in enumerate(ROOM_COLORS)
        if np.any(labels == index)
    ]
    graph.add_nodes_from(types)
    for left_index, left_name in enumerate(ROOM_COLORS):
        left = labels == left_index
        if not left.any():
            continue
        dilated = binary_dilation(left, iterations=3)
        for right_index, right_name in enumerate(ROOM_COLORS):
            if right_index <= left_index:
                continue
            if np.any(dilated & (labels == right_index)):
                graph.add_edge(left_name, right_name)
    return graph


def graph_metrics(
    target_after: np.ndarray,
    predicted_after: np.ndarray,
) -> dict[str, float]:
    predicted = _type_graph(predicted_after)
    target = _type_graph(target_after)
    node_intersection = len(set(predicted) & set(target))
    node_precision = node_intersection / len(predicted) if predicted else 0.0
    node_recall = node_intersection / len(target) if target else 0.0
    node_f1 = (
        2.0 * node_precision * node_recall / (node_precision + node_recall)
        if node_precision + node_recall
        else 1.0
    )
    pred_edges = {tuple(sorted(edge)) for edge in predicted.edges}
    target_edges = {tuple(sorted(edge)) for edge in target.edges}
    union = pred_edges | target_edges
    edge_overlap = (
        len(pred_edges & target_edges) / len(union)
        if union
        else 1.0
    )
    return {
        "node_f1": float(node_f1),
        "edge_overlap": float(edge_overlap),
        "ged": float(
            nx.graph_edit_distance(
                predicted,
                target,
                node_subst_cost=lambda left, right: int(left != right),
                node_del_cost=lambda _node: 1,
                node_ins_cost=lambda _node: 1,
            )
        ),
    }

"""Hard-constraint, intent-preservation, and topology utility metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage


def _mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def intent_preservation(
    prediction: np.ndarray,
    control: np.ndarray,
    taxonomy,
) -> dict[str, float]:
    room_ids = np.asarray(taxonomy.room_ids)
    control_rooms = np.isin(control, room_ids)
    prediction_rooms = np.isin(prediction, room_ids)
    room_pixels = int(control_rooms.sum())
    preserved_room_pixels = int(
        np.logical_and(control_rooms, prediction == control).sum()
    )
    return {
        "control_room_pixel_preservation": (
            preserved_room_pixels / room_pixels if room_pixels else 1.0
        ),
        "control_room_support_drift": 1.0
        - _mask_iou(prediction_rooms, control_rooms),
        "control_wall_preservation": _mask_iou(
            prediction == taxonomy.class_id("wall"),
            control == taxonomy.class_id("wall"),
        ),
        "control_opening_preservation": _mask_iou(
            np.isin(prediction, taxonomy.opening_ids),
            np.isin(control, taxonomy.opening_ids),
        ),
        "control_furniture_preservation": _mask_iou(
            np.isin(prediction, taxonomy.furniture_ids),
            np.isin(control, taxonomy.furniture_ids),
        ),
    }


def room_component_topology(
    labels: np.ndarray,
    taxonomy,
    *,
    opening_radius: int = 2,
) -> dict[str, object]:
    """Extract room components and door-mediated room adjacency."""

    room_ids = np.asarray(taxonomy.room_ids)
    components = np.zeros(labels.shape, dtype=np.int32)
    room_components = 0
    connectivity = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for room_id in room_ids:
        room_mask = labels == room_id
        room_labels, count = ndimage.label(room_mask, structure=connectivity)
        if count:
            components[room_mask] = room_labels[room_mask] + room_components
            room_components += int(count)
    if room_components == 0:
        return {
            "room_components": 0,
            "door_contact_room_fraction": 1.0,
            "door_graph_components": 0,
            "largest_room_group_fraction": 0.0,
            "room_adjacency_edges": 0,
            "room_adjacency": set(),
        }

    opening_mask = np.isin(labels, taxonomy.opening_ids)
    opening_components, opening_count = ndimage.label(
        opening_mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    edges: set[tuple[int, int]] = set()
    touched: set[int] = set()
    for opening_id in range(1, int(opening_count) + 1):
        coordinates = np.argwhere(opening_components == opening_id)
        if not len(coordinates):
            continue
        low = np.maximum(coordinates.min(axis=0) - opening_radius, 0)
        high = np.minimum(
            coordinates.max(axis=0) + opening_radius + 1,
            np.asarray(labels.shape),
        )
        slice_y = slice(int(low[0]), int(high[0]))
        slice_x = slice(int(low[1]), int(high[1]))
        nearby = np.unique(components[slice_y, slice_x])
        room_ids_near = [int(value) for value in nearby if value > 0]
        if not room_ids_near:
            continue
        touched.update(room_ids_near)
        for left_index, left in enumerate(room_ids_near):
            for right in room_ids_near[left_index + 1 :]:
                edges.add(tuple(sorted((left, right))))

    groups = _component_groups(room_components, edges)
    largest_group = max((len(group) for group in groups), default=0)
    return {
        "room_components": int(room_components),
        "door_contact_room_fraction": len(touched) / room_components,
        "door_graph_components": len(groups),
        "largest_room_group_fraction": largest_group / room_components,
        "room_adjacency_edges": len(edges),
        "room_adjacency": edges,
    }


def _component_groups(
    component_count: int,
    edges: Iterable[tuple[int, int]],
) -> list[set[int]]:
    adjacency = {index: set() for index in range(1, component_count + 1)}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    groups: list[set[int]] = []
    remaining = set(adjacency)
    while remaining:
        start = remaining.pop()
        group = {start}
        frontier = [start]
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency[node]:
                if neighbor not in group:
                    group.add(neighbor)
                    remaining.discard(neighbor)
                    frontier.append(neighbor)
        groups.append(group)
    return groups


def adjacency_f1(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy,
) -> dict[str, float]:
    predicted = room_component_topology(prediction, taxonomy)
    expected = room_component_topology(target, taxonomy)
    predicted_edges: set[tuple[int, int]] = predicted["room_adjacency"]
    expected_edges: set[tuple[int, int]] = expected["room_adjacency"]
    shared = len(predicted_edges & expected_edges)
    precision = shared / len(predicted_edges) if predicted_edges else 0.0
    recall = shared / len(expected_edges) if expected_edges else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "room_adjacency_precision": precision,
        "room_adjacency_recall": recall,
        "room_adjacency_f1": f1,
    }


__all__ = [
    "adjacency_f1",
    "intent_preservation",
    "room_component_topology",
]

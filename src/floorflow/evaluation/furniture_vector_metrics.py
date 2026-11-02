"""Metrics for room-attached vector furniture instances."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, label
from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import ReadoutFurniture


def _polygon_mask(
    polygon: Sequence[Sequence[float]],
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.rint(np.asarray(polygon)).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _majority_name(values: np.ndarray, taxonomy: FloorplanTaxonomy) -> str | None:
    if values.size == 0:
        return None
    counts = np.bincount(values.reshape(-1), minlength=taxonomy.num_classes)
    class_id = int(counts.argmax())
    if counts[class_id] == 0:
        return None
    return taxonomy.class_names[class_id]


def _majority_room_name(
    values: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> str | None:
    if values.size == 0:
        return None
    counts = np.bincount(values.reshape(-1), minlength=taxonomy.num_classes)
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    room_counts = counts[room_ids]
    if int(room_counts.max()) == 0:
        return None
    return taxonomy.class_names[int(room_ids[int(room_counts.argmax())])]


def furniture_vector_metrics(
    furniture: Sequence[ReadoutFurniture],
    rooms: Sequence[dict],
    target_labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    """Measure geometry, label, attachment, and containment quality."""

    target_labels = np.asarray(target_labels, dtype=np.int64)
    shape = tuple(target_labels.shape)
    predicted_mask = np.zeros(shape, dtype=bool)
    predicted_room_mask = np.zeros(shape, dtype=bool)
    target_instances = []
    for class_id in taxonomy.furniture_ids:
        components, count = label(
            target_labels == int(class_id),
            structure=np.asarray(
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                dtype=np.uint8,
            ),
        )
        for component_id in range(1, int(count) + 1):
            target_instances.append(
                (
                    components == component_id,
                    taxonomy.class_names[int(class_id)],
                )
            )
    valid_polygons = 0
    label_matches = 0
    matched_label_matches = 0
    matched_target_ious = []
    attachment_matches = 0
    attachment_count = 0
    containment_values = []
    for item in furniture:
        polygon = Polygon(item.polygon)
        valid_polygons += int(polygon.is_valid and not polygon.is_empty)
        mask = _polygon_mask(item.polygon, shape)
        if not mask.any():
            continue
        predicted_mask |= mask
        predicted_label = _majority_name(target_labels[mask], taxonomy)
        label_matches += int(predicted_label == item.label)
        best_iou = 0.0
        best_label = None
        for target_instance, target_label in target_instances:
            intersection = float((mask & target_instance).sum())
            union = float((mask | target_instance).sum())
            iou = intersection / max(1.0, union)
            if iou > best_iou:
                best_iou = iou
                best_label = target_label
        if best_iou > 0.0:
            matched_label_matches += int(best_label == item.label)
            matched_target_ious.append(best_iou)
        if 0 <= item.room_index < len(rooms):
            room = rooms[item.room_index]
            predicted_room_mask |= _polygon_mask(room["polygon"], shape)
            local_ring = binary_dilation(mask, iterations=3) & ~mask
            expected_room = _majority_room_name(
                target_labels[local_ring],
                taxonomy,
            )
            attachment_matches += int(expected_room == room["label"])
            attachment_count += 1
            room_polygon = Polygon(room["polygon"])
            if polygon.is_valid and room_polygon.is_valid and polygon.area > 0.0:
                containment_values.append(
                    polygon.intersection(room_polygon).area / polygon.area
                )
    target_mask = np.isin(target_labels, np.asarray(taxonomy.furniture_ids))
    intersection = int((predicted_mask & target_mask).sum())
    denominator = int(predicted_mask.sum() + target_mask.sum())
    union = int((predicted_mask | target_mask).sum())
    count = max(1, len(furniture))
    return {
        "furniture_vector_precision": (
            intersection / max(1, int(predicted_mask.sum()))
        ),
        "furniture_vector_recall": (
            intersection / max(1, int(target_mask.sum()))
        ),
        "furniture_vector_f1": (
            2.0 * intersection / denominator
            if denominator
            else 1.0
        ),
        "furniture_vector_iou": (
            intersection / union
            if union
            else 1.0
        ),
        "furniture_vector_count": float(len(furniture)),
        "furniture_vector_polygon_valid_rate": valid_polygons / count,
        "furniture_vector_label_accuracy": label_matches / count,
        "furniture_vector_matched_label_accuracy": (
            matched_label_matches / count
        ),
        "furniture_vector_matched_target_iou": (
            float(np.mean(matched_target_ious))
            if matched_target_ious
            else 0.0
        ),
        "furniture_vector_attachment_accuracy": (
            attachment_matches / attachment_count
            if attachment_count
            else 0.0
        ),
        "furniture_vector_containment": (
            float(np.mean(containment_values))
            if containment_values
            else 1.0
        ),
        "furniture_vector_out_of_boundary": (
            float((predicted_mask & ~predicted_room_mask).sum())
            / max(1, int(predicted_mask.sum()))
        ),
    }


__all__ = ["furniture_vector_metrics"]

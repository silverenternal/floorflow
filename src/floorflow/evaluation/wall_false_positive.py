"""Classify predicted wall false positives by geometric provenance."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from floorflow.data.taxonomy import FloorplanTaxonomy


def wall_false_positive_decomposition(
    prediction: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    wall_tolerance: float = 2.0,
    boundary_tolerance: float = 2.0,
    exterior_tolerance: float = 3.0,
) -> dict[str, float | int]:
    """Split false wall pixels into extension, boundary, and unsupported bins."""

    predicted = np.asarray(prediction)
    expected = np.asarray(target)
    if predicted.shape != expected.shape:
        raise ValueError("prediction and target must share a shape")
    area = np.asarray(footprint, dtype=bool)
    if area.ndim == 3:
        area = area[0]
    if predicted.shape != area.shape:
        raise ValueError("footprint must match the label map")
    wall_id = taxonomy.class_id("wall")
    predicted_wall = (predicted == wall_id) & area
    target_wall = (expected == wall_id) & area
    false_positive = predicted_wall & ~target_wall
    false_negative = target_wall & ~predicted_wall
    true_positive = predicted_wall & target_wall

    target_rooms = np.isin(expected, np.asarray(taxonomy.room_ids))
    room_support_boundary = target_rooms & ~binary_erosion(
        target_rooms,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    room_labels = np.where(target_rooms, expected, 0)
    horizontal = np.zeros_like(target_rooms)
    vertical = np.zeros_like(target_rooms)
    horizontal[:, 1:] = room_labels[:, 1:] != room_labels[:, :-1]
    vertical[1:, :] = room_labels[1:, :] != room_labels[:-1, :]
    room_boundary = room_support_boundary | horizontal | vertical
    footprint_boundary = area & ~binary_erosion(
        area,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    near_wall = distance_transform_edt(~target_wall) <= float(
        wall_tolerance
    )
    near_room_boundary = distance_transform_edt(~room_boundary) <= float(
        boundary_tolerance
    )
    near_exterior = distance_transform_edt(~footprint_boundary) <= float(
        exterior_tolerance
    )

    target_wall_extension = false_positive & near_wall
    remaining = false_positive & ~target_wall_extension
    exterior_duplication = remaining & near_exterior
    remaining = remaining & ~exterior_duplication
    room_boundary_fragment = remaining & near_room_boundary
    unsupported_free = remaining & ~room_boundary_fragment
    counts = {
        "true_positive": int(true_positive.sum()),
        "false_positive": int(false_positive.sum()),
        "false_negative": int(false_negative.sum()),
        "target_wall_extension": int(target_wall_extension.sum()),
        "room_boundary_fragment": int(room_boundary_fragment.sum()),
        "exterior_duplication": int(exterior_duplication.sum()),
        "unsupported_free": int(unsupported_free.sum()),
    }
    denominator = max(1, counts["false_positive"])
    return {
        "wall_false_positive_pixels": counts["false_positive"],
        "wall_false_negative_pixels": counts["false_negative"],
        "wall_true_positive_pixels": counts["true_positive"],
        "wall_target_extension_rate": (
            counts["target_wall_extension"] / denominator
        ),
        "wall_room_boundary_fragment_rate": (
            counts["room_boundary_fragment"] / denominator
        ),
        "wall_exterior_duplication_rate": (
            counts["exterior_duplication"] / denominator
        ),
        "wall_unsupported_free_rate": (
            counts["unsupported_free"] / denominator
        ),
        "wall_false_positive_per_target_wall_pixel": (
            counts["false_positive"] / max(1, counts["true_positive"])
        ),
    }


__all__ = ["wall_false_positive_decomposition"]

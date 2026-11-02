"""Lightweight raster-to-vector opening diagnostics for training validation."""

from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.opening_readout import (
    detections_from_labels,
    opening_detection_metrics,
)
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan


def opening_vector_proxy(
    predicted: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    tolerance: float = 4.0,
) -> dict[str, float]:
    """Decode labels into vector openings and return canonical-style matching."""

    predicted_plan = raster_labels_to_wall_plan(predicted, taxonomy)
    target_plan = raster_labels_to_wall_plan(target, taxonomy)
    predicted_detections = detections_from_labels(
        predicted,
        taxonomy,
        predicted_plan,
    )
    target_detections = detections_from_labels(
        target,
        taxonomy,
        target_plan,
    )
    metrics = opening_detection_metrics(
        predicted_detections,
        target_detections,
        tolerance=float(tolerance),
    )
    return {
        "opening_vector_precision_proxy": float(metrics["precision"]),
        "opening_vector_recall_proxy": float(metrics["recall"]),
        "opening_vector_f1_proxy": float(metrics["f1"]),
        "opening_vector_predicted_count_proxy": float(
            len(predicted_detections)
        ),
        "opening_vector_target_count_proxy": float(
            len(target_detections)
        ),
        "opening_vector_geometry_matched_proxy": float(
            metrics["geometry_matched"]
        ),
        "opening_vector_family_consistent_rate_proxy": float(
            metrics["family_consistent"]
        )
        / max(1.0, float(metrics["geometry_matched"])),
        "opening_vector_center_error_proxy": float(
            metrics["geometry_center_error_mean"]
        ),
        "opening_vector_width_ratio_proxy": float(
            metrics["geometry_width_ratio_mean"]
        ),
        "opening_vector_door_to_window_proxy": float(
            metrics["door_to_window"]
        ),
        "opening_vector_window_to_door_proxy": float(
            metrics["window_to_door"]
        ),
    }


__all__ = ["opening_vector_proxy"]

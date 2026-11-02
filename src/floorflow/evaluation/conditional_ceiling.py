"""Contour-neighbour diagnostics for room-layout identifiability."""

from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy


def room_macro_iou_numpy(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> float:
    values = []
    for class_id in taxonomy.room_ids:
        expected = target == class_id
        if not expected.any():
            continue
        predicted = prediction == class_id
        intersection = np.logical_and(predicted, expected).sum()
        union = np.logical_or(predicted, expected).sum()
        values.append(float(intersection / max(1, union)))
    return float(np.mean(values)) if values else 0.0


def distribution_entropy(distribution: np.ndarray) -> float:
    values = np.asarray(distribution, dtype=np.float64)
    values = values[values > 0.0]
    return float(-(values * np.log(values)).sum())


def neighbour_iou_summary(
    prediction_masks: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    """Summarise neighbour targets without selecting a model output."""

    ious = np.asarray(
        [
            room_macro_iou_numpy(prediction, target, taxonomy)
            for prediction in prediction_masks
        ],
        dtype=np.float64,
    )
    return {
        "top1_iou": float(ious[0]),
        "neighbour_mean_iou": float(ious.mean()),
        "neighbour_median_iou": float(np.median(ious)),
        "neighbour_max_iou": float(ious.max()),
        "neighbour_at_least_0_95": float((ious >= 0.95).mean()),
    }

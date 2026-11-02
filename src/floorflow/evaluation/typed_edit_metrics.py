"""Metrics for matched typed-edit routing policies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import binary_dilation

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.edit_detection import edit_detection_iou_mse
from floorflow.evaluation.housemind_structure import housemind_style_metrics
from floorflow.inference.topology_scaffold import verify_topology


def normalized_region_mask(
    shape: tuple[int, int],
    normalized: Sequence[float],
) -> np.ndarray:
    """Rasterize a clamped normalized edit rectangle."""

    x, y, width, height = normalized
    rows, columns = shape
    left = max(0, min(columns, round(float(x) * columns)))
    top = max(0, min(rows, round(float(y) * rows)))
    right = max(left + 1, min(columns, round((x + width) * columns)))
    bottom = max(top + 1, min(rows, round((y + height) * rows)))
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def mask_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    if not mask.any():
        return 0.0
    return float(
        np.square(
            prediction[mask].astype(np.float32)
            - target[mask].astype(np.float32)
        ).mean()
    )


def typed_edit_metrics(
    composition,
    before: dict,
    target_after: np.ndarray,
    region: np.ndarray,
    target_watertight: bool,
    taxonomy: FloorplanTaxonomy,
) -> dict:
    """Score one routed output under the shared protocol."""

    footprint = before["footprint"].astype(bool)
    wall = (
        np.isin(before["target"], [taxonomy.class_id("wall")])
        | np.isin(target_after, [taxonomy.class_id("wall")])
    )
    wall = binary_dilation(wall, iterations=1)
    valid = footprint & ~wall
    outside = valid & ~region
    inside = valid & region
    visible = composition.visible
    outside_pixels = int(outside.sum())
    outside_changes = int(
        np.count_nonzero(visible[outside] != before["prediction"][outside])
    )
    pixel_exact = (
        float((visible[outside] == before["prediction"][outside]).mean())
        if outside.any()
        else 1.0
    )
    room_exact = (
        float(
            (
                composition.room_labels[outside]
                == before["room_labels"][outside]
            ).mean()
        )
        if outside.any()
        else 1.0
    )
    delta_iou, delta_mse = edit_detection_iou_mse(
        before["target"],
        target_after,
        before["prediction"],
        visible,
        valid,
    )
    validity = verify_topology(composition.topology, taxonomy)
    structural = housemind_style_metrics(
        visible,
        target_after,
        footprint,
        taxonomy,
        ged_timeout_seconds=0.5,
    )
    outside_mse = mask_mse(visible, target_after, outside)
    before_outside_mse = mask_mse(before["prediction"], target_after, outside)
    return {
        "outside_pixel_exact": pixel_exact,
        "outside_room_exact": room_exact,
        "outside_changes": outside_changes,
        "outside_valid_pixels": outside_pixels,
        "locality_violation_rate": (
            float(outside_changes / outside_pixels)
            if outside_pixels
            else 0.0
        ),
        "outside_delta_mse": outside_mse - before_outside_mse,
        "outside_mse_to_target": outside_mse,
        "delta_iou": float(delta_iou),
        "delta_mse": float(delta_mse),
        "validity": float(bool(validity.get("valid", False))),
        "conditional_watertight": float(
            bool(validity.get("watertight", False))
            if target_watertight
            else True
        ),
        "node_f1": float(structural["node_f1"]),
        "node_precision": float(structural["node_precision"]),
        "node_recall": float(structural["node_recall"]),
        "edge_overlap": float(structural["edge_overlap"]),
        "edge_precision": float(structural["edge_precision"]),
        "edge_recall": float(structural["edge_recall"]),
        "edge_f1": float(structural["edge_f1"]),
        "ged": float(structural["ged"]),
        "exact_ged": float(structural["exact"]),
        "inside_change_fraction": (
            float(
                (
                    visible[inside]
                    != before["prediction"][inside]
                ).mean()
            )
            if inside.any()
            else 0.0
        ),
        "routing_fraction": float(composition.routing_mask.mean()),
        "outside_region_changes_precontract": (
            composition.outside_region_changes
        ),
    }


__all__ = [
    "mask_mse",
    "normalized_region_mask",
    "typed_edit_metrics",
]

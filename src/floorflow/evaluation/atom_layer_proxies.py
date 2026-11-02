"""Cheap layer proxies that track final vector CAD metrics."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_dilation, label

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.opening_vector_proxy import opening_vector_proxy
from floorflow.evaluation.wall_metrics import wall_fields


def _component_count(mask: np.ndarray) -> int:
    components, _count = label(
        np.asarray(mask, dtype=bool),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(components.ravel())[1:]
    return int((sizes >= 4).sum())


def layer_proxy_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    predicted_room_support: np.ndarray | None = None,
) -> dict[str, float]:
    """Return low-cost layer diagnostics used only after target-free inference."""

    predicted = np.asarray(predicted)
    target = np.asarray(target)
    footprint = np.asarray(footprint, dtype=bool)
    prediction_tensor = torch.from_numpy(predicted)[None].long()
    target_tensor = torch.from_numpy(target)[None].long()
    wall = wall_fields(prediction_tensor, target_tensor, taxonomy)
    wall_id = taxonomy.class_id("wall")
    opening_ids = np.asarray(taxonomy.opening_ids)
    furniture_ids = np.asarray(taxonomy.furniture_ids)
    room_ids = np.asarray(taxonomy.room_ids)
    predicted_opening = np.isin(predicted, opening_ids) & footprint
    target_opening = np.isin(target, opening_ids) & footprint
    target_wall_band = binary_dilation(
        target == wall_id,
        iterations=2,
    )
    predicted_furniture = np.isin(predicted, furniture_ids) & footprint
    target_furniture = np.isin(target, furniture_ids) & footprint
    predicted_rooms = (
        np.asarray(predicted_room_support, dtype=bool)
        if predicted_room_support is not None
        else np.isin(predicted, room_ids)
    ) & footprint
    metrics = {
        "wall_centerline_f1_proxy": float(wall["wall_centerline_f1"]),
        "wall_iou_proxy": float(wall["wall_iou"]),
        "wall_distance_mae_proxy": float(wall["wall_distance_mae"]),
        "opening_wall_alignment_proxy": (
            float((predicted_opening & target_wall_band).sum())
            / max(1, int(predicted_opening.sum()))
        ),
        "opening_target_band_recall_proxy": (
            float((target_opening & target_wall_band).sum())
            / max(1, int(target_opening.sum()))
        ),
        "opening_count_abs_error_proxy": abs(
            _component_count(predicted_opening)
            - _component_count(target_opening)
        ),
        "furniture_count_abs_error_proxy": abs(
            _component_count(predicted_furniture)
            - _component_count(target_furniture)
        ),
        "furniture_predicted_room_containment_proxy": (
            float((predicted_furniture & predicted_rooms).sum())
            / max(1, int(predicted_furniture.sum()))
        ),
    }
    metrics.update(
        opening_vector_proxy(predicted, target, taxonomy)
    )
    return metrics


__all__ = ["layer_proxy_metrics"]

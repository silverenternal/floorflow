"""HouseMind-style edit-detection metrics for paired floorplan edits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from floorflow.data.control_cache import read_control_cache
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.control_baseline import control_layers
from floorflow.representation.layered_floorplan import LayeredFloorplan


def decode_visible_controls(
    path: Path,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    controls = read_control_cache(path)
    footprint_tensor = torch.from_numpy(footprint)
    rooms, wall, openings, furniture = control_layers(
        controls,
        footprint_tensor,
        taxonomy,
    )
    return LayeredFloorplan(
        room_labels=rooms[None],
        wall=wall.long()[None],
        opening=openings[None],
        furniture=furniture[None],
        footprint=footprint_tensor[None],
        taxonomy=taxonomy,
    ).visible_labels()[0].numpy()


def edit_detection_iou_mse(
    target_before: np.ndarray,
    target_after: np.ndarray,
    prediction_before: np.ndarray,
    prediction_after: np.ndarray,
    valid_region: np.ndarray,
) -> tuple[float, float]:
    expected = (target_before != target_after) & valid_region
    predicted = (prediction_before != prediction_after) & valid_region
    union = np.logical_or(expected, predicted).sum()
    iou = (
        1.0
        if union == 0
        else float(np.logical_and(expected, predicted).sum() / union)
    )
    mse = float(
        np.square(expected.astype(np.float32) - predicted.astype(np.float32))[
            valid_region
        ].mean()
    )
    return iou, mse


__all__ = ["decode_visible_controls", "edit_detection_iou_mse"]

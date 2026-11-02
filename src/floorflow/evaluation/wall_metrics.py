"""Wall-specific structural metrics and optional centerline proxy."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from floorflow.data.taxonomy import FloorplanTaxonomy


def thin_binary_mask(mask: np.ndarray, max_iterations: int = 128) -> np.ndarray:
    """Zhang-Suen thinning for a stable wall-centreline proxy."""

    image = np.asarray(mask, dtype=np.uint8).copy()
    for _ in range(max_iterations):
        changed = False
        for phase in (0, 1):
            padded = np.pad(image, 1)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = np.zeros_like(image, dtype=np.uint8)
            ring = (p2, p3, p4, p5, p6, p7, p8, p9)
            for index, current in enumerate(ring):
                following = ring[(index + 1) % len(ring)]
                transitions += ((current == 0) & (following == 1)).astype(np.uint8)
            if phase == 0:
                condition = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                condition = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (image == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & condition
            if remove.any():
                image[remove] = 0
                changed = True
        if not changed:
            break
    return image.astype(bool)


def _centerline_f1(predicted: np.ndarray, expected: np.ndarray, tolerance: int = 2) -> float:
    predicted_line = thin_binary_mask(predicted)
    expected_line = thin_binary_mask(expected)
    if not predicted_line.any() and not expected_line.any():
        return 1.0
    if not predicted_line.any() or not expected_line.any():
        return 0.0
    predicted_near = binary_dilation(predicted_line, iterations=tolerance)
    expected_near = binary_dilation(expected_line, iterations=tolerance)
    precision = (predicted_line & expected_near).sum() / max(1, predicted_line.sum())
    recall = (expected_line & predicted_near).sum() / max(1, expected_line.sum())
    return float(2.0 * precision * recall / max(1.0e-8, precision + recall))


def wall_fields(
    prediction: torch.Tensor,
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    wall_id = taxonomy.class_id("wall")
    predicted = (prediction == wall_id).float()
    expected = (target == wall_id).float()
    intersection = (predicted * expected).sum()
    union = (predicted + expected - predicted * expected).sum().clamp_min(1.0)
    distance_pred = distance_transform_edt((predicted[0].cpu().numpy() == 0).astype(np.uint8))
    distance_target = distance_transform_edt((expected[0].cpu().numpy() == 0).astype(np.uint8))
    distance_mae = float(np.abs(distance_pred - distance_target).mean())
    opening_ids = np.asarray(taxonomy.opening_ids)
    structural_predicted = (
        predicted[0].bool().cpu().numpy()
        | np.isin(prediction[0].cpu().numpy(), opening_ids)
    )
    components, _ = label(
        structural_predicted,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(components.ravel())[1:]
    closure = (
        float(sizes.max() / max(1, structural_predicted.sum()))
        if sizes.size
        else 1.0
    )
    centerline_f1 = _centerline_f1(
        predicted[0].cpu().numpy().astype(bool),
        expected[0].cpu().numpy().astype(bool),
    )
    return {
        "wall_iou": float((intersection / union).detach().cpu()),
        "wall_distance_mae": distance_mae,
        "wall_closure": float(closure),
        "wall_centerline_f1": centerline_f1,
    }

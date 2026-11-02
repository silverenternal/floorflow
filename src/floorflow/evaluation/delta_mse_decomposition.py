"""Decompose HouseMind Delta-MSE into structural and interior regions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.ndimage import binary_dilation

from floorflow.data.taxonomy import FloorplanTaxonomy


def class_boundary(labels: np.ndarray) -> np.ndarray:
    """Return pixels adjacent to a categorical label change."""
    if labels.ndim != 2:
        raise ValueError("labels must be a two-dimensional label map")
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def structural_boundary_band(
    before: np.ndarray,
    after: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    iterations: int = 2,
) -> np.ndarray:
    """Boundary band where walls or openings touch another label."""
    structural = np.asarray(
        [
            index
            for index, name in enumerate(taxonomy.class_names)
            if name == "wall" or taxonomy.is_opening(name)
        ],
        dtype=np.uint8,
    )
    structural_mask = np.isin(before, structural) | np.isin(after, structural)
    boundary = class_boundary(before) | class_boundary(after)
    boundary &= binary_dilation(structural_mask, iterations=1)
    return binary_dilation(boundary, iterations=iterations)


def room_boundary_band(
    before: np.ndarray,
    after: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    iterations: int = 2,
) -> np.ndarray:
    """Boundary band between adjacent room labels."""
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.uint8)
    before_rooms = np.isin(before, room_ids)
    after_rooms = np.isin(after, room_ids)
    boundaries = class_boundary(before) | class_boundary(after)
    boundaries &= before_rooms | after_rooms
    return binary_dilation(boundaries, iterations=iterations)


def decomposition_masks(
    target_before: np.ndarray,
    target_after: np.ndarray,
    valid: np.ndarray,
    edit_region: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, np.ndarray]:
    """Return mutually exclusive masks for edit-region error analysis."""
    structure = structural_boundary_band(
        target_before,
        target_after,
        taxonomy,
    )
    rooms = room_boundary_band(
        target_before,
        target_after,
        taxonomy,
    )
    structure &= edit_region & valid
    rooms &= edit_region & valid & ~structure
    interior = edit_region & valid & ~structure & ~rooms
    return {
        "edit_region": edit_region & valid,
        "structure_boundary": structure,
        "room_boundary": rooms,
        "edit_interior": interior,
        "outside": valid & ~edit_region,
        "global": valid,
    }


def _mean_on_mask(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return float(selected.mean()) if selected.size else None


def decompose_delta_error(
    target_before: np.ndarray,
    target_after: np.ndarray,
    prediction_before: np.ndarray,
    prediction_after: np.ndarray,
    valid: np.ndarray,
    edit_region: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, dict[str, float | int | None]]:
    """Compute regional Delta-MSE and error-mass diagnostics."""
    expected = (target_before != target_after) & valid
    predicted = (prediction_before != prediction_after) & valid
    errors = np.square(expected.astype(np.float32) - predicted.astype(np.float32))
    masks = decomposition_masks(
        target_before,
        target_after,
        valid,
        edit_region,
        taxonomy,
    )
    edit_error_mass = float(errors[masks["edit_region"]].sum())
    total_pixels = max(1, int(valid.sum()))
    results = {}
    for name, mask in masks.items():
        pixels = int(mask.sum())
        mass = float(errors[mask].sum())
        results[name] = {
            "pixels": pixels,
            "area_fraction_of_valid": float(pixels / total_pixels),
            "mse": _mean_on_mask(errors, mask),
            "error_mass": mass,
            "error_mass_fraction_of_edit": (
                float(mass / edit_error_mass) if edit_error_mass > 0.0 else 0.0
            ),
        }
    return results


def average_decompositions(
    rows: Iterable[dict[str, dict[str, float | int | None]]],
) -> dict[str, dict[str, float]]:
    rows = list(rows)
    result = {}
    for region in rows[0]:
        region_rows = [
            row[region]
            for row in rows
            if int(row[region]["pixels"] or 0) > 0
        ]
        result[region] = {
            key: float(np.mean([row[key] for row in region_rows]))
            for key in (
                "area_fraction_of_valid",
                "mse",
                "error_mass_fraction_of_edit",
            )
        }
        result[region]["cases"] = float(len(region_rows))
    return result


__all__ = [
    "average_decompositions",
    "class_boundary",
    "decompose_delta_error",
    "decomposition_masks",
    "room_boundary_band",
    "structural_boundary_band",
]

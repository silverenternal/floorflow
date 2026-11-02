"""Target-free structural features for guarded repair selection."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label

from floorflow.data.taxonomy import FloorplanTaxonomy


def target_free_repair_features(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    """Features available at inference, without reference labels."""

    footprint = np.asarray(footprint, dtype=bool)
    interior = binary_erosion(footprint, iterations=3) if footprint.any() else footprint
    rooms = np.isin(labels, taxonomy.room_ids)
    wall = labels == taxonomy.class_id("wall")
    openings = np.isin(labels, taxonomy.opening_ids)
    furniture = np.isin(labels, taxonomy.furniture_ids)
    coverage = (rooms & interior).sum() / max(1, interior.sum())
    opening_alignment = (
        (openings & binary_dilation(wall, iterations=2)).sum() / max(1, openings.sum())
        if openings.any()
        else 1.0
    )
    containment = (
        (furniture & binary_dilation(rooms, iterations=2)).sum() / max(1, furniture.sum())
        if furniture.any()
        else 1.0
    )
    _, room_components = label(rooms)
    _, wall_components = label(wall)
    empty_fraction = ((labels == 0) & interior).sum() / max(1, interior.sum())
    return {
        "feature_room_coverage": float(coverage),
        "feature_opening_alignment": float(opening_alignment),
        "feature_containment": float(containment),
        "feature_room_fragments": float(room_components / max(1, len(taxonomy.room_ids))),
        "feature_wall_fragments": float(wall_components / max(1, interior.sum()) * 1000.0),
        "feature_empty_fraction": float(empty_fraction),
    }


def repair_decision_features(
    raw: np.ndarray,
    candidate: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    """Raw/candidate/delta features available before accepting a repair."""

    raw_features = target_free_repair_features(raw, footprint, taxonomy)
    candidate_features = target_free_repair_features(candidate, footprint, taxonomy)
    output = {}
    for name, value in raw_features.items():
        base = name.removeprefix("feature_")
        output[f"feature_raw_{base}"] = value
        output[f"feature_candidate_{base}"] = candidate_features[name]
        output[f"feature_delta_{base}"] = candidate_features[name] - value
    return output

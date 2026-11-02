"""Tests for deterministic Raster2Seq polygon fusion."""

from __future__ import annotations

import numpy as np

from floorflow.evaluation.raster2seq_polygon_fusion import (
    FusionConfig,
    align_to_reference,
    canonical_points,
    fuse_cluster,
    fuse_selected_view,
    resample,
    signed_area,
)


def square(offset: float = 0.0) -> dict:
    return {
        "category_id": 1,
        "segmentation": [
            2.0 + offset,
            2.0,
            8.0 + offset,
            2.0,
            8.0 + offset,
            8.0,
            2.0 + offset,
            8.0,
        ],
    }


def test_resample_is_closed_shape_and_positive_orientation():
    points = canonical_points(square())
    sampled = resample(points, 32)
    assert sampled.shape == (32, 2)
    assert signed_area(sampled) > 0.0


def test_cyclic_alignment_recovers_shifted_contour():
    reference = resample(canonical_points(square()), 32)
    shifted = np.roll(reference, 7, axis=0)
    aligned = align_to_reference(shifted, reference)
    assert np.allclose(aligned, reference)


def test_fused_cluster_uses_fallback_for_single_view():
    original = square()
    cluster = {"items": [("original", original)], "sources": {"original"}}
    fused = fuse_cluster(
        cluster,
        original,
        config=FusionConfig(),
    )
    assert fused is original


def test_selected_view_fusion_preserves_room_count():
    original = square()
    shifted = square(0.25)
    items = {
        "original": [original],
        "hflip": [{**shifted, "id": 1}],
    }
    clusters = [
        {
            "items": [("original", original), ("hflip", items["hflip"][0])],
            "sources": {"original", "hflip"},
        }
    ]
    fused = fuse_selected_view(
        items,
        "original",
        clusters,
        frozenset(),
        FusionConfig(),
    )
    assert len(fused) == 1
    assert fused[0]["category_id"] == 1

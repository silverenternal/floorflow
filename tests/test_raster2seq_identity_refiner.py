"""Tests for conservative Raster2Seq identity refinement."""

from __future__ import annotations

from floorflow.inference.raster2seq_identity_refiner import (
    Raster2SeqRefineConfig,
    fuse_raster2seq_items,
    refine_raster2seq_items,
)


def _room(identifier: int, points: list[list[float]], category: int = 1) -> dict:
    return {
        "image_id": "00000",
        "segmentation": points,
        "category_id": category,
        "id": identifier,
    }


def test_identity_config_returns_exact_items() -> None:
    items = [
        _room(0, [[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]]),
        _room(1, [[0.0, 8.0], [8.0, 8.0]], category=10),
    ]
    refined, stats = refine_raster2seq_items(items, Raster2SeqRefineConfig())
    assert refined == items
    assert stats["removed_rooms"] == 0


def test_shared_axis_snapping_and_duplicate_removal() -> None:
    items = [
        _room(0, [[0.0, 0.0], [0.0, 5.0], [5.0, 5.0], [5.0, 0.0]]),
        _room(1, [[5.4, 0.0], [5.4, 5.0], [10.0, 5.0], [10.0, 0.0]]),
        _room(2, [[0.1, 0.1], [0.1, 4.9], [4.9, 4.9], [4.9, 0.1]]),
        _room(3, [[0.0, 8.0], [8.0, 8.0]], category=9),
    ]
    refined, stats = refine_raster2seq_items(
        items,
        Raster2SeqRefineConfig(
            simplify_tolerance=0.2,
            snap_tolerance=0.5,
            duplicate_iou=0.95,
        ),
    )
    shared = refined[0]["segmentation"][2][0]
    assert abs(shared - refined[1]["segmentation"][0][0]) < 1.0e-9
    assert stats["removed_rooms"] == 1
    assert refined[-1]["category_id"] == 9


def test_tiny_room_can_be_removed_without_touching_openings() -> None:
    items = [
        _room(0, [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.5, 0.0]]),
        _room(1, [[2.0, 2.0], [2.0, 4.0]], category=9),
    ]
    refined, stats = refine_raster2seq_items(
        items,
        Raster2SeqRefineConfig(minimum_room_area=1.0),
    )
    assert [item["category_id"] for item in refined] == [9]
    assert stats["removed_rooms"] == 1


def test_ensemble_union_keeps_complementary_rooms() -> None:
    primary = [_room(0, [[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]])]
    secondary = [
        _room(0, [[0.1, 0.0], [0.1, 4.0], [4.0, 4.0], [4.0, 0.0]]),
        _room(1, [[4.0, 0.0], [4.0, 4.0], [8.0, 4.0], [8.0, 0.0]]),
    ]
    fused, stats = fuse_raster2seq_items(
        primary,
        secondary,
        mode="union",
        match_iou=0.8,
    )
    assert stats["output_items"] == 2
    assert fused[-1]["id"] == 1

"""Tests for the HouseMind-style free-text editor."""

from __future__ import annotations

import numpy as np

from floorflow.evaluation.housemind_free_text_metrics import edit_metrics
from floorflow.inference.housemind_free_text_editor import (
    ROOM_COLORS,
    edit_layout,
    parse_instruction,
)


def canvas() -> np.ndarray:
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[4:28, 4:28] = ROOM_COLORS["LivingRoom"]
    image[4:12, 4:12] = ROOM_COLORS["Bathroom"]
    image[3:29, 3] = 0
    image[3:29, 28] = 0
    return image


def test_parser_handles_english_and_chinese_directions():
    english = parse_instruction("Add a small Bathroom in the northwest.")
    chinese = parse_instruction("删除在西区域的阳台。")
    assert english is not None and english.operation == "add"
    assert english.room_type == "Bathroom" and english.direction == "northwest"
    assert chinese is not None and chinese.operation == "delete"
    assert chinese.room_type == "Balcony" and chinese.direction == "west"


def test_delete_fills_target_with_adjacent_room():
    image = canvas()
    request = parse_instruction("Delete the Bathroom.")
    assert request is not None
    edited, diagnostics = edit_layout(image, request)
    assert diagnostics["applied"]
    assert np.all(edited[5:11, 5:11] == ROOM_COLORS["LivingRoom"])


def test_identical_edit_has_zero_error():
    image = canvas()
    metrics = edit_metrics(image, image, image)
    assert metrics["delta_iou"] == 1.0
    assert metrics["delta_mse"] == 0.0

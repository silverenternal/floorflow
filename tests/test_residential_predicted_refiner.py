"""Tests for predicted-control refiner evaluation metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_residential_predicted_refiner",
    ROOT / "scripts/train_residential_predicted_refiner.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_room_and_support_iou_are_perfect_for_identical_labels() -> None:
    target = np.zeros((8, 8), dtype=np.int64)
    target[:4] = 2
    target[4:] = 3
    room_ids = [2, 3]
    assert MODULE._room_iou(target, target, room_ids) == 1.0
    assert MODULE._support_iou(target, target, room_ids) == 1.0


def test_support_iou_ignores_background_class() -> None:
    target = np.zeros((4, 4), dtype=np.int64)
    target[:2] = 2
    prediction = target.copy()
    prediction[2:] = 99
    assert MODULE._support_iou(prediction, target, [2]) == 1.0

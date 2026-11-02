from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.external_baseline_metrics import (
    HOUSE_ROOM_COLORS,
    evaluate_housegan_cases,
)


class _Dataset:
    def __init__(self, taxonomy, target, footprint):
        self.taxonomy = taxonomy
        self.target = target
        self.footprint = footprint

    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return self.target, self.footprint, 1, "case"


def test_housegan_metrics_separate_class_iou_and_footprint_violation(tmp_path: Path) -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"LivingRoom": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    target = np.zeros((16, 16), dtype=np.uint8)
    living = taxonomy.class_id("room:LivingRoom")
    target[2:10, 2:10] = living
    footprint = np.zeros((16, 16), dtype=bool)
    footprint[1:12, 1:12] = True
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    color = np.asarray(
        [int(HOUSE_ROOM_COLORS[1][i : i + 2], 16) for i in (1, 3, 5)],
        dtype=np.uint8,
    )
    image[2:9, 2:9] = color
    Image.fromarray(image).save(tmp_path / "fp_final_0.png")
    dataset = _Dataset(taxonomy, target, footprint)

    report = evaluate_housegan_cases(dataset, tmp_path)

    assert report["cases"] == 1
    assert report["condition_mode"] == "oracle_room_program"
    assert report["summary"]["outside_footprint_pixels_mean"] == 0
    assert 0.0 < report["summary"]["mapped_class_iou_mean"] < 1.0

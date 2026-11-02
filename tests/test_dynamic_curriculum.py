from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.dynamic_curriculum import (
    DataDynamicsTracker,
    data_map_weights,
    load_data_map,
    write_data_map,
)
from floorflow.data.taxonomy import build_taxonomy


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Bed": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_data_dynamics_tracker_records_family_and_room_metrics(tmp_path) -> None:
    taxonomy = _taxonomy()
    names = ("case-a__Floor-1", "case-b__Floor-1")
    tracker = DataDynamicsTracker(names, taxonomy)
    target = torch.zeros((2, 8, 8), dtype=torch.long)
    target[:, 1:4, 1:4] = taxonomy.class_id("room:Bedroom")
    target[:, 5:7, 5:7] = taxonomy.class_id("door:Swing Beside")
    good = torch.nn.functional.one_hot(
        target,
        taxonomy.num_classes,
    ) * 20
    bad = torch.zeros_like(good)
    bad[..., taxonomy.class_id("wall")] = 20
    mixed = torch.stack((good[0], bad[1]))
    footprint = torch.ones((2, 1, 8, 8))
    tracker.update(names, good.permute(0, 3, 1, 2), target, footprint)
    tracker.update(names, mixed.permute(0, 3, 1, 2), target, footprint)
    points = tracker.snapshot()
    assert points[0].visits == 2
    assert points[0].confidence > 0.99
    assert points[0].room_recall == 1.0
    assert points[0].room_macro_iou == 1.0
    assert points[1].confidence < 0.51
    assert points[1].room_recall == 0.5
    assert 0.49 < points[1].room_macro_iou < 0.51
    assert points[1].forgetfulness == 1

    weights = data_map_weights(points, strength=0.5)
    assert set(weights) == set(names)
    assert weights[names[1]] > weights[names[0]]

    summary = write_data_map(points, tmp_path / "data_map.jsonl", epoch=1)
    assert summary["samples"] == 2
    assert summary["regions"]
    assert (tmp_path / "data_map.jsonl").is_file()
    restored = load_data_map(tmp_path / "data_map.jsonl")
    assert [point.name for point in restored] == list(names)


def test_data_map_waits_for_two_visits() -> None:
    taxonomy = _taxonomy()
    names = ("case-a__Floor-1",)
    tracker = DataDynamicsTracker(names, taxonomy)
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    logits = torch.zeros((1, taxonomy.num_classes, 8, 8))
    tracker.update(names, logits, target, torch.ones((1, 1, 8, 8)))
    points = tracker.snapshot()
    assert points[0].region == "warming"
    assert data_map_weights(points)[names[0]] == 1.0


def test_task_focus_prioritizes_weak_room_recall() -> None:
    taxonomy = _taxonomy()
    names = ("case-a__Floor-1", "case-b__Floor-1")
    tracker = DataDynamicsTracker(names, taxonomy)
    target = torch.zeros((2, 8, 8), dtype=torch.long)
    target[:, 1:4, 1:4] = taxonomy.class_id("room:Bedroom")
    good = torch.nn.functional.one_hot(target, taxonomy.num_classes) * 20
    weak = good.clone()
    weak[1, 1:4, 1:4] = 0
    weak[1, 1:4, 1:4, 0] = 20
    footprint = torch.ones((2, 1, 8, 8))
    tracker.update(names, good.permute(0, 3, 1, 2), target, footprint)
    tracker.update(names, weak.permute(0, 3, 1, 2), target, footprint)
    points = tracker.snapshot()
    weights = data_map_weights(points, strength=0.5, task_focus=2.0)
    assert weights[names[1]] > weights[names[0]]

    guarded = data_map_weights(
        points,
        strength=0.5,
        task_focus=2.0,
        eligible={names[0]: True, names[1]: False},
    )
    assert guarded[names[1]] == 1.0

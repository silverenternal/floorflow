from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from floorflow.data.taxonomy import build_taxonomy
from floorflow.training import room_sampling
from floorflow.training.room_sampling import (
    build_room_curriculum_sampler,
    prepare_raster_dataset,
    raster_dataset_kwargs,
)


class _Dataset:
    def __init__(self) -> None:
        self.samples = [
            SimpleNamespace(relative_case=f"case_{index}", floor_id="Floor-1")
            for index in range(4)
        ]
        self._targets = None
        self._footprints = None

    def preload(self) -> None:
        self._targets = []
        self._footprints = []
        for index in range(4):
            target = np.zeros((16, 16), dtype=np.int64)
            target[1 : 3 + index, 1 : 4] = 1
            target[5 : 7 + index, 5 : 8] = 2
            self._targets.append(target)
            self._footprints.append(np.ones((16, 16), dtype=bool))


def test_room_curriculum_sampler_covers_complexity_pool_deterministically() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "LivingRoom": 4}),
            "furniture": Counter(),
            "door": Counter({"Swing Beside": 1}),
            "window": Counter({"Regular Window": 1}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    dataset = _Dataset()
    sampler = build_room_curriculum_sampler(
        dataset,
        taxonomy,
        num_samples=4,
        seed=7,
    )
    sampler.set_epoch(0, 4)
    first = list(sampler)
    sampler.set_epoch(0, 4)
    second = list(sampler)
    assert first == second
    assert len(first) == 4


def test_prepare_raster_dataset_selects_disjoint_offset_slice(monkeypatch) -> None:
    monkeypatch.setattr(
        room_sampling,
        "filter_unsupervised_samples",
        lambda dataset: [],
    )
    dataset = _Dataset()
    removed = prepare_raster_dataset(
        dataset,
        limit=2,
        offset=1,
        shuffle_seed=7,
    )
    assert removed == []
    assert len(dataset.samples) == 2
    repeated = _Dataset()
    prepare_raster_dataset(
        repeated,
        limit=2,
        offset=1,
        shuffle_seed=7,
    )
    selected_names = [
        f"{sample.relative_case}__{sample.floor_id}"
        for sample in dataset.samples
    ]
    repeated_names = [
        f"{sample.relative_case}__{sample.floor_id}"
        for sample in repeated.samples
    ]
    assert selected_names == repeated_names


def test_prepare_raster_dataset_rejects_short_slice(monkeypatch) -> None:
    monkeypatch.setattr(
        room_sampling,
        "filter_unsupervised_samples",
        lambda dataset: [],
    )
    with pytest.raises(ValueError, match="only 1 valid samples"):
        prepare_raster_dataset(_Dataset(), limit=2, offset=3)


def test_raster_dataset_kwargs_precomputes_room_targets_in_workers() -> None:
    args = SimpleNamespace(
        teacher_only=False,
        queries=24,
        control_raster_root=None,
        page_floor_crop=True,
    )
    assert raster_dataset_kwargs(args) == {
        "image_mode": "floor",
        "include_room_targets": True,
        "room_target_limit": 24,
    }
    args.teacher_only = True
    assert raster_dataset_kwargs(args)["include_room_targets"] is False

from __future__ import annotations

import numpy as np

from floorflow.representation.room_pair_prior import (
    accumulate_pair_counts,
    normalize_pair_counts,
)


class _Taxonomy:
    room_ids = (10, 20)


def test_pair_counts_record_adjacent_room_classes(monkeypatch) -> None:
    target = np.zeros((4, 4), dtype=np.uint8)
    target[:, :2] = 10
    target[:, 2:] = 20
    footprint = np.ones((4, 4), dtype=np.uint8)
    import floorflow.representation.room_pair_prior as module

    monkeypatch.setattr(
        module,
        "build_room_partition_target",
        lambda labels, mask, taxonomy: np.where(labels == 10, 0, 1),
    )
    counts = accumulate_pair_counts(
        np.zeros((2, 2)),
        target,
        footprint,
        _Taxonomy(),
    )
    assert counts[0, 1] == 4
    prior = normalize_pair_counts(counts, smoothing=0.0)
    assert prior[0, 1] == 1.0
    assert prior[1, 0] == 1.0

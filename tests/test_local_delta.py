"""Tests for strict local and typed floorplan edit composition."""

from __future__ import annotations

import numpy as np

from floorflow.inference.local_delta import compose_typed_delta


def test_compose_typed_delta_changes_only_typed_region():
    before = np.ones((3, 3), dtype=np.uint8)
    before_control = np.array(
        [[1, 9, 1], [1, 1, 1], [1, 1, 1]],
        dtype=np.uint8,
    )
    after_control = np.array(
        [[1, 2, 1], [1, 1, 1], [1, 1, 1]],
        dtype=np.uint8,
    )
    edit_region = np.array(
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        dtype=bool,
    )

    repaired_before, repaired_after = compose_typed_delta(
        before,
        before_control,
        after_control,
        edit_region,
    )

    assert repaired_before[0, 1] == 9
    assert repaired_after[0, 1] == 2
    outside = ~edit_region
    assert np.array_equal(repaired_before[outside], before[outside])
    assert np.array_equal(repaired_after[outside], before[outside])

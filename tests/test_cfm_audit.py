from __future__ import annotations

import torch

from floorflow.distribution.cfm_audit import (
    condition_cache_reuse_ratio,
    family_stream_distinctness,
    outside_support_change,
)


def test_cfm_audit_detects_outside_change_and_stream_separation() -> None:
    before = torch.zeros(1, 2, 4, 4)
    after = before.clone()
    after[:, :, 0, 0] = 1.0
    footprint = torch.zeros(1, 1, 4, 4)
    footprint[:, :, 1:3, 1:3] = 1.0
    assert outside_support_change(before, after, footprint) > 0.0
    streams = {
        "room": torch.zeros(1, 2, 4),
        "wall": torch.ones(1, 2, 4),
    }
    assert family_stream_distinctness(streams)["room_wall"] > 0.0
    assert condition_cache_reuse_ratio(8) == 0.875

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.predicted_operation_success import (
    control_region_mask,
    operation_success,
    room_control_request,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 3, "Kitchen": 3}),
            "furniture": Counter({"Sink": 3}),
            "door": Counter({"Swing": 3}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_control_region_mask_is_clamped_and_nonempty() -> None:
    outside = control_region_mask((-5, -4, 3, 2), (10, 8))
    assert outside[0:1, 0:1].all()
    assert outside.sum() == 1

    clipped = control_region_mask((8, 7, 5, 5), (10, 8))
    assert clipped[7, 8:10].all()
    assert clipped.sum() == 2


def test_room_control_request_uses_signed_logit_deltas() -> None:
    taxonomy = _taxonomy()
    source = torch.zeros(len(taxonomy.room_ids), 4, 4)
    edited = source.clone()
    source[0, 1:3, 1:3] = 4.0
    edited[0, 1:3, 1:3] = -4.0
    edited[1, 1:3, 1:3] = 4.0
    request = room_control_request(
        {"room_logits": source},
        {"room_logits": edited},
        (1, 1, 2, 2),
        taxonomy,
    )
    assert request.source_class_id == taxonomy.room_ids[0]
    assert request.target_class_id == taxonomy.room_ids[1]


def test_room_type_success_requires_target_and_source_change() -> None:
    taxonomy = _taxonomy()
    source = taxonomy.room_ids[0]
    target = taxonomy.room_ids[1]
    before = np.full((8, 8), source)
    after = before.copy()
    region = np.zeros_like(before, dtype=bool)
    region[2:6, 2:6] = True
    after[region] = target
    result = operation_success(
        before,
        after,
        region,
        taxonomy,
        "room_type",
        request=_request(source, target),
    )
    assert result.success
    assert result.source_removed_fraction == 1.0
    assert result.target_share_of_changed == 1.0

    retained = before.copy()
    result = operation_success(
        before,
        retained,
        region,
        taxonomy,
        "room_type",
        request=_request(source, target),
    )
    assert not result.success


def test_room_remove_requires_source_removal() -> None:
    taxonomy = _taxonomy()
    source = taxonomy.room_ids[0]
    target = taxonomy.room_ids[1]
    before = np.full((8, 8), source)
    after = before.copy()
    region = np.zeros_like(before, dtype=bool)
    region[2:6, 2:6] = True
    after[region] = target
    result = operation_success(
        before,
        after,
        region,
        taxonomy,
        "room_remove",
        request=_request(source, target),
    )
    assert result.success


def test_object_delete_success_requires_requested_family() -> None:
    taxonomy = _taxonomy()
    opening = taxonomy.opening_ids[0]
    before = np.zeros((8, 8), dtype=np.int64)
    before[3:5, 3:5] = opening
    after = before.copy()
    region = np.zeros((8, 8), dtype=bool)
    region[2:6, 2:6] = True

    failed = operation_success(
        before,
        after,
        region,
        taxonomy,
        "opening_delete",
    )
    assert not failed.success

    after[region] = 0
    passed = operation_success(
        before,
        after,
        region,
        taxonomy,
        "opening_delete",
    )
    assert passed.success


def _request(source: int, target: int):
    from floorflow.evaluation.predicted_operation_success import RoomControlRequest

    return RoomControlRequest(
        source_class_id=int(source),
        target_class_id=int(target),
        source_logit_delta=-12.0,
        target_logit_delta=12.0,
    )

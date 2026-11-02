from __future__ import annotations

import torch

from floorflow.chip_flow.structural_relations import structural_relation_features


def _constraints(cluster_ids: list[int]) -> torch.Tensor:
    constraints = torch.zeros(1, len(cluster_ids), 5)
    constraints[0, :, 3] = torch.tensor(cluster_ids, dtype=torch.float32)
    return constraints


def test_group_width_uses_member_bbox_span_not_mean_member_width() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [10.0, 0.0, 1.0, 1.0]]]
    )
    areas = boxes[..., 2] * boxes[..., 3]
    features, active = structural_relation_features(
        boxes,
        areas,
        _constraints([1, 1]),
    )
    scale = areas.sum(dim=1).sqrt()
    torch.testing.assert_close(
        features[0, 0, 2] * scale[0],
        torch.tensor(11.0),
    )
    torch.testing.assert_close(
        features[0, 0, 0] * scale[0],
        torch.tensor(5.5),
    )
    assert active.tolist() == [[True, False, False, False, False, False, False, False, False]]


def test_group_span_is_permutation_invariant_and_masked_slots_stay_zero() -> None:
    boxes = torch.tensor(
        [[[20.0, 4.0, 2.0, 3.0], [1.0, 2.0, 4.0, 1.0]]]
    )
    areas = boxes[..., 2] * boxes[..., 3]
    constraints = _constraints([2, 2])
    features, active = structural_relation_features(boxes, areas, constraints)
    permuted, permuted_active = structural_relation_features(
        boxes.flip(1),
        areas.flip(1),
        constraints.flip(1),
    )
    torch.testing.assert_close(permuted, features)
    torch.testing.assert_close(permuted_active, active)
    assert torch.count_nonzero(features[~active]) == 0

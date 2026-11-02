from __future__ import annotations

import torch

from floorflow.objectives.room_query_matching import (
    RoomQueryMatch,
    match_room_mask_queries,
    match_room_queries,
)
from floorflow.objectives.room_query_separation import (
    supervised_same_class_overlap_loss,
)


def test_room_query_matching_uses_class_and_polygon_cost() -> None:
    classes = torch.tensor(
        [
            [
                [8.0, 0.0],
                [0.0, 8.0],
                [1.0, 3.0],
            ]
        ]
    )
    polygons = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2]],
                [[0.8, 0.8], [1.0, 0.8], [1.0, 1.0]],
                [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6]],
            ]
        ]
    )
    target_classes = torch.tensor([[1, 0]])
    target_polygons = torch.tensor(
        [
            [
                [[0.8, 0.8], [1.0, 0.8], [1.0, 1.0]],
                [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2]],
            ]
        ]
    )
    valid = torch.tensor([[True, True]])
    match = match_room_queries(
        classes,
        polygons,
        target_classes,
        target_polygons,
        valid,
    )[0]
    assert match.predicted_indices.tolist() == [0, 1]
    assert match.target_indices.tolist() == [1, 0]


def test_room_query_matching_can_use_joint_mask_identity() -> None:
    classes = torch.zeros(1, 2, 2)
    polygons = torch.zeros(1, 2, 1, 2)
    target_classes = torch.tensor([[0, 1]])
    target_polygons = torch.zeros(1, 2, 1, 2)
    predicted_masks = torch.zeros(1, 2, 4, 4)
    predicted_masks[0, 0, :, 2:] = 8.0
    predicted_masks[0, 1, :, :2] = 8.0
    target_masks = torch.zeros(1, 2, 4, 4)
    target_masks[0, 0, :, :2] = 1.0
    target_masks[0, 1, :, 2:] = 1.0

    match = match_room_queries(
        classes,
        polygons,
        target_classes,
        target_polygons,
        torch.tensor([[True, True]]),
        target_counts=[2],
        predicted_masks=predicted_masks,
        target_masks=target_masks,
        mask_cost_weight=10.0,
    )[0]
    assert set(
        zip(
            match.predicted_indices.tolist(),
            match.target_indices.tolist(),
        )
    ) == {(0, 1), (1, 0)}


def test_room_mask_matching_uses_one_shared_class_and_mask_identity() -> None:
    class_logits = torch.tensor(
        [[[16.0, 0.0, 0.0], [0.0, 16.0, 0.0], [0.0, 0.0, 16.0]]]
    )
    target_classes = torch.tensor([[0, 1]])
    target_masks = torch.zeros(1, 2, 4, 4)
    target_masks[0, 0, :, :2] = 1.0
    target_masks[0, 1, :, 2:] = 1.0
    mask_logits = torch.stack(
        [
            torch.full((4, 4), 8.0),
            torch.full((4, 4), -8.0),
            torch.full((4, 4), -8.0),
        ]
    )[None]
    mask_logits[0, 0, :, :2] = -8.0
    mask_logits[0, 1, :, 2:] = 8.0

    matches = match_room_mask_queries(
        class_logits,
        mask_logits,
        target_classes,
        target_masks,
        torch.tensor([[True, True]]),
        class_cost_weight=100.0,
    )[0]

    assert set(
        zip(
            matches.predicted_indices.tolist(),
            matches.target_indices.tolist(),
        )
    ) == {(0, 0), (1, 1)}


def test_supervised_separation_only_uses_same_class_positive_pairs() -> None:
    mask_logits = torch.full((1, 3, 4, 4), 12.0)
    matches = [
        RoomQueryMatch(
            predicted_indices=torch.tensor([0, 1]).numpy(),
            target_indices=torch.tensor([0, 1]).numpy(),
        )
    ]
    same = supervised_same_class_overlap_loss(
        mask_logits,
        matches,
        torch.tensor([[1, 1]]),
    )
    different = supervised_same_class_overlap_loss(
        mask_logits,
        matches,
        torch.tensor([[1, 2]]),
    )
    assert torch.isfinite(same)
    assert same > different

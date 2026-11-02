from __future__ import annotations

import torch

from floorflow.objectives.room_query_separation import (
    active_query_overlap_loss,
    active_query_presence,
    balanced_room_class_weights,
    count_gated_query_separation,
    gated_active_query_presence,
)


def test_duplicate_room_queries_have_higher_overlap_loss() -> None:
    presence = torch.ones(1, 3)
    disjoint = torch.full((1, 3, 4, 4), -12.0)
    disjoint[0, 0, 0:2, 0:2] = 12.0
    disjoint[0, 1, 0:2, 2:4] = 12.0
    disjoint[0, 2, 2:4, :] = 12.0
    duplicate = disjoint.clone()
    duplicate[0, 1] = duplicate[0, 0]
    assert active_query_overlap_loss(disjoint, presence) < active_query_overlap_loss(
        duplicate,
        presence,
    )


def test_overlap_loss_is_finite_and_differentiable() -> None:
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    presence = torch.rand(2, 4)
    loss = active_query_overlap_loss(logits, presence)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_class_aware_overlap_penalizes_cross_class_pairs_less() -> None:
    logits = torch.full((1, 2, 4, 4), 12.0)
    presence = torch.ones(1, 2)
    same_class = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    different_class = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    same = active_query_overlap_loss(
        logits,
        presence,
        class_probability=same_class,
    )
    different = active_query_overlap_loss(
        logits,
        presence,
        class_probability=different_class,
    )
    assert same > different


def test_presence_and_balanced_weights_follow_identity_contract() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 0.0],
                [0.0, 4.0],
                [2.0, 1.0],
            ]
        ]
    )
    presence = active_query_presence(logits)
    assert presence[0, 0] > 0.95
    assert presence[0, 1] < 0.05
    classes = torch.tensor([[0, 0, 1, 1]])
    valid = torch.tensor([[True, True, True, False]])
    weights = balanced_room_class_weights(classes, valid, 3)
    assert weights[0] < weights[1]
    assert weights[2] == 1.0


def test_query_separation_presence_uses_count_selection_gate() -> None:
    classes = torch.tensor([[[4.0, 0.0], [4.0, 0.0]]])
    existence = torch.tensor([[8.0, 8.0]])
    selection = torch.tensor([[True, False]])
    presence = gated_active_query_presence(classes, existence, selection)
    assert presence[0, 0] > 0.95
    assert presence[0, 1] == 0.0
    masks = torch.zeros(1, 2, 4, 4)
    positive = count_gated_query_separation(
        classes,
        masks,
        existence,
        torch.tensor([[1.0, 1.0]]),
    )
    suppressed = count_gated_query_separation(
        classes,
        masks,
        existence,
        selection,
    )
    assert positive > 0.0
    assert suppressed == 0.0

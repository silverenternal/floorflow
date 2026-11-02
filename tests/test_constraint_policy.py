"""Tests for model-origin action policy objectives."""

from __future__ import annotations

import torch

from floorflow.objectives.constraint_policy import (
    hard_masked_best_action_loss,
    hard_masked_utility_loss,
    hard_masked_utility_pairwise_loss,
    utility_target_distribution,
)


def test_utility_target_assigns_zero_mass_to_invalid_candidates() -> None:
    utility = torch.tensor([[0.8, 0.4, 0.9]])
    hard = torch.tensor([[True, False, True]])
    target = utility_target_distribution(utility, hard)
    assert target[0, 1] == 0.0
    assert torch.allclose(target.sum(dim=-1), torch.ones(1))


def test_ranking_losses_are_finite_and_receive_gradients() -> None:
    logits = torch.randn(2, 4, requires_grad=True)
    utility = torch.tensor(
        [
            [0.8, 0.4, 0.2, 0.1],
            [0.7, 0.6, 0.3, 0.2],
        ]
    )
    hard = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )
    loss = hard_masked_utility_loss(logits, utility, hard)
    pairwise = hard_masked_utility_pairwise_loss(logits, utility, hard)
    assert torch.isfinite(loss)
    assert torch.isfinite(pairwise)
    (loss + pairwise).backward()
    assert logits.grad is not None


def test_best_action_loss_rewards_unique_legal_maximum() -> None:
    hard = torch.tensor([[True, True, False]])
    utility = torch.tensor([[0.7, 0.2, 9.0]])
    good = torch.tensor([[2.0, 0.0, -5.0]], requires_grad=True)
    bad = torch.tensor([[0.0, 2.0, -5.0]], requires_grad=True)
    good_loss = hard_masked_best_action_loss(good, utility, hard)
    bad_loss = hard_masked_best_action_loss(bad, utility, hard)
    assert good_loss < bad_loss

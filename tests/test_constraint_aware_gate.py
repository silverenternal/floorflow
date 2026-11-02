"""Tests for hard-masked constraint-aware candidate gating."""

from __future__ import annotations

import numpy as np
import torch

from floorflow.evaluation.constraint_ranking import ndcg_at_k
from floorflow.model.constraint_aware_gate import (
    ConstraintAwareGate,
    hard_masked_log_softmax,
)
from floorflow.objectives.constraint_gate_loss import (
    gate_calibration_loss,
    hard_masked_listwise_loss,
    hard_masked_pairwise_loss,
)


def test_gate_assigns_exact_zero_to_invalid_candidates() -> None:
    model = ConstraintAwareGate(
        hidden_dim=32,
        action_features=15,
        hard_mask_features=8,
    )
    nodes = 24
    output = model(
        torch.randn(2, nodes, 8),
        torch.eye(nodes)[None].repeat(2, 1, 1),
        torch.zeros(2, nodes, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.randn(2, 15),
        torch.tensor(
            [
                [1, 0, 1, 0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        output["gate_probability"][~output["hard_feasible"]],
        torch.zeros_like(output["gate_probability"][~output["hard_feasible"]]),
    )
    log_probability = hard_masked_log_softmax(
        output["gate_logits"],
        output["hard_feasible"],
    )
    assert torch.equal(
        log_probability[~output["hard_feasible"]],
        torch.full_like(log_probability[~output["hard_feasible"]], -torch.inf),
    )


def test_full_gate_is_in_parameter_budget() -> None:
    model = ConstraintAwareGate()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 2_000_000 <= parameters <= 5_000_000


def test_gate_losses_are_finite_with_partial_hard_masks() -> None:
    logits = torch.randn(2, 5, requires_grad=True)
    hard = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    positive = torch.tensor(
        [
            [True, True, False, False, False],
            [True, False, True, False, False],
        ]
    )
    listwise = hard_masked_listwise_loss(logits, positive, hard)
    pairwise = hard_masked_pairwise_loss(logits, positive, hard)
    calibration = gate_calibration_loss(
        logits,
        positive.float(),
        hard,
    )
    assert torch.isfinite(listwise)
    assert torch.isfinite(pairwise)
    assert torch.isfinite(calibration)


def test_ndcg_at_k_prefers_relevant_candidates() -> None:
    relevance = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    good = ndcg_at_k(np.array([0.9, 0.8, 0.1, 0.0, 0.0]), relevance)
    bad = ndcg_at_k(np.array([0.1, 0.0, 0.9, 0.8, 0.7]), relevance)
    assert good == 1.0
    assert 0.0 < bad < good

"""Tests for plan-conditioned interaction residuals."""

from __future__ import annotations

import torch

from floorflow.model.interaction_constraint_gate import (
    InteractionConstraintAwareGate,
)


def _inputs():
    batch, nodes = 2, 5
    return {
        "node_features": torch.randn(batch, nodes, 8),
        "adjacency": torch.eye(nodes)[None].expand(batch, -1, -1),
        "family_ids": torch.zeros(batch, nodes, dtype=torch.long),
        "operation_ids": torch.zeros(batch, dtype=torch.long),
        "action_features": torch.randn(batch, 15),
        "hard_mask": torch.ones(batch, 8, dtype=torch.bool),
        "global_features": torch.randn(batch, 15),
    }


def test_plan_context_changes_candidate_scores_but_not_terminal_policy() -> None:
    torch.manual_seed(7)
    model = InteractionConstraintAwareGate(
        hidden_dim=32,
        embedding_dim=16,
        sequence_hidden_dim=16,
        interaction_scale=1.0,
    ).eval()
    with torch.no_grad():
        model.sequence_score_head.weight.fill_(0.1)
        model.sequence_score_head.bias.zero_()
    inputs = _inputs()
    sequence = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0],
        ]
    )
    with torch.no_grad():
        base = model.base(**inputs)
        planned = model(**inputs, sequence_features=sequence)
    assert not torch.allclose(
        planned["gate_logits"][0],
        base["gate_logits"][0],
    )
    assert torch.allclose(
        planned["gate_logits"][1],
        base["gate_logits"][1],
    )
    assert planned["gate_logits"].requires_grad == base["gate_logits"].requires_grad

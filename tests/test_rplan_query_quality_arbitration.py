from __future__ import annotations

import pytest
import torch

from floorflow.inference.rplan_query_quality_arbitration import (
    RPLAN_QUERY_QUALITY_FEATURE_NAMES,
    make_rplan_query_quality_head,
    rplan_query_quality_features,
    score_rplan_query_quality,
    select_quality_queries,
)


def _predictions() -> dict[str, torch.Tensor]:
    return {
        "room_instance_class_logits": torch.randn(2, 4, 4),
        "room_exists": torch.randn(2, 4),
        "room_instance_mask_logits": torch.randn(2, 4, 8, 8),
        "room_polygons": torch.rand(2, 4, 12, 2),
        "room_adjacency_logits": torch.randn(2, 4, 4),
        "routing": {
            "routing_entropy": torch.tensor(0.5),
        },
    }


def test_query_quality_features_are_target_free_and_finite() -> None:
    features = rplan_query_quality_features(_predictions())
    assert features.shape == (2, 4, len(RPLAN_QUERY_QUALITY_FEATURE_NAMES))
    assert torch.isfinite(features).all()


def test_query_quality_head_scores_each_query_and_backpropagates() -> None:
    features = rplan_query_quality_features(_predictions()).requires_grad_(True)
    head = make_rplan_query_quality_head(hidden_size=16)
    score = score_rplan_query_quality(head, features)
    score.mean().backward()
    assert score.shape == (2, 4)
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_quality_selection_preserves_query_budget_and_minimum() -> None:
    existence = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
    quality = torch.tensor([[4.0, -4.0, 3.0, -4.0]])
    selected = select_quality_queries(
        existence,
        quality,
        torch.tensor([3]),
        minimum_quality=0.2,
    )
    assert selected.tolist() == [[True, False, True, False]]


def test_quality_selection_keeps_one_query_when_all_scores_are_low() -> None:
    selected = select_quality_queries(
        torch.tensor([[3.0, 2.0, 1.0]]),
        torch.full((1, 3), -8.0),
        torch.tensor([2]),
        minimum_quality=0.2,
    )
    assert selected.tolist() == [[True, False, False]]
    with pytest.raises(ValueError, match="share \\[B,Q\\]"):
        select_quality_queries(
            torch.zeros(1, 3),
            torch.zeros(1, 2),
            torch.tensor([1]),
        )

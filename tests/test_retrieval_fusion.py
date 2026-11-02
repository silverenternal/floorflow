from __future__ import annotations

import torch

from floorflow.model.retrieval_fusion import LearnedRetrievalFusion


def test_retrieval_fusion_is_identity_at_initialization() -> None:
    fusion = LearnedRetrievalFusion(5, hidden_channels=16, gate_resolution=8)
    model_logits = torch.randn(2, 5, 16, 16)
    retrieval_logits = torch.randn(2, 5, 16, 16)
    fused, diagnostics = fusion(
        model_logits,
        retrieval_logits,
        torch.ones(2, 1, 16, 16),
    )
    assert torch.allclose(fused, model_logits)
    assert diagnostics["gate_active_fraction"] == 0


def test_retrieval_fusion_gate_receives_gradient() -> None:
    fusion = LearnedRetrievalFusion(
        5,
        hidden_channels=16,
        gate_resolution=8,
        base_prior_weight=0.75,
    )
    model_logits = torch.randn(1, 5, 16, 16)
    retrieval_logits = torch.randn(1, 5, 16, 16)
    fused, diagnostics = fusion(
        model_logits,
        retrieval_logits,
        torch.ones(1, 1, 16, 16),
    )
    assert not torch.allclose(fused, model_logits)
    assert diagnostics["base_prior_weight"].item() == 0.75
    diagnostics["gate"].mean().backward()
    assert fusion.gate[-1].weight.grad is not None
    assert fusion.gate[-1].weight.grad.abs().sum() > 0

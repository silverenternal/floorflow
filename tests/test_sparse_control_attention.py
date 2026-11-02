from __future__ import annotations

import torch

from floorflow.model.official_mita_route import OfficialMiTARouteAttention
from floorflow.model.sparse_control_attention import SparseControlAttention


def test_sparse_control_attention_shapes_and_diagnostics() -> None:
    attention = SparseControlAttention(32, token_grid=4, router_count=4, top_k=2)
    assert isinstance(attention.attention, OfficialMiTARouteAttention)
    attention.scale.data.fill_(1.0)
    state = torch.randn(2, 32, 16, 16)
    control = torch.randn(2, 32, 16, 16)
    mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    mask[:, :, :4] = True
    footprint = torch.ones(2, 1, 16, 16)
    output, diagnostics = attention(state, control, mask, footprint)
    assert output.shape == state.shape
    assert 0.0 <= float(diagnostics["mask_coverage"]) <= 1.0
    assert torch.isfinite(diagnostics["routing_entropy"])
    assert torch.isfinite(diagnostics["router_prior_abs"])
    assert 1.0 <= float(diagnostics["router_budget_mean"])
    assert float(diagnostics["router_budget_mean"]) <= 3.0


def test_sparse_control_attention_mha_path() -> None:
    attention = SparseControlAttention(32, attention_type="mha", token_grid=4)
    output, diagnostics = attention(
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 32, 8, 8),
        torch.zeros(1, 1, 8, 8, dtype=torch.bool),
        torch.ones(1, 1, 8, 8),
    )
    assert output.shape == (1, 32, 8, 8)
    assert "mask_coverage" in diagnostics


def test_adaptive_control_attention_routes_on_visible_coverage() -> None:
    torch.manual_seed(3)
    attention = SparseControlAttention(
        32,
        attention_type="adaptive",
        token_grid=4,
        router_count=4,
        top_k=2,
    )
    state = torch.randn(2, 32, 16, 16)
    control = torch.rand(2, 32, 16, 16)
    footprint = torch.ones(2, 1, 16, 16)
    full_mask = torch.zeros_like(footprint)
    full_mask[:, :, 5:11, 5:11] = 1.0
    sparse_mask = torch.zeros_like(footprint)
    sparse_mask[:, :, 6:7, 6:7] = 1.0
    output, diagnostics = attention(
        state,
        control,
        torch.stack([full_mask[0], sparse_mask[0]]),
        footprint,
    )
    assert output.shape == state.shape
    assert 0.0 < diagnostics["adaptive_mha_weight"] < 1.0
    per_sample = diagnostics["adaptive_mha_weight_per_sample"]
    assert per_sample[0] < per_sample[1]
    output.square().mean().backward()
    assert attention.adaptive_scale.grad is not None

from __future__ import annotations

import torch

from floorflow.model.official_mita_route import OfficialMiTARouteAttention


def test_official_route_attention_shapes_and_gradients() -> None:
    attention = OfficialMiTARouteAttention(
        32,
        num_heads=4,
        router_count=9,
        top_k=5,
    )
    query = torch.randn(2, 16, 32, requires_grad=True)
    memory = torch.randn(2, 20, 32, requires_grad=True)
    output, diagnostics = attention(query, memory)
    assert output.shape == query.shape
    assert diagnostics["router_usage"].shape == (9,)
    assert int(diagnostics["top_k"]) == 5
    output.mean().backward()
    assert query.grad is not None and memory.grad is not None


def test_official_route_excludes_agent_key_path() -> None:
    attention = OfficialMiTARouteAttention(
        16,
        num_heads=2,
        router_count=4,
        top_k=3,
    )
    query = torch.randn(1, 4, 16)
    memory = torch.randn(1, 8, 16)
    with torch.no_grad():
        first, _ = attention(query, memory)
        attention.v_proj.weight.zero_()
        attention.v_proj.bias.zero_()
        attention.out_proj.bias.zero_()
        zero_value, _ = attention(query, memory)
    assert torch.allclose(zero_value, torch.zeros_like(first), atol=1.0e-6)


def test_official_route_accepts_dynamic_budgets_and_prior() -> None:
    attention = OfficialMiTARouteAttention(
        16,
        num_heads=2,
        router_count=4,
        top_k=4,
    )
    query = torch.randn(2, 6, 16)
    memory = torch.randn(2, 9, 16)
    prior = torch.randn(2, 6, 4, requires_grad=True)
    output, diagnostics = attention(
        query,
        memory,
        router_prior=prior,
        router_budgets=torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
    )
    assert torch.isfinite(output).all()
    assert float(diagnostics["router_budget_min"]) == 1.0
    assert float(diagnostics["router_budget_max"]) == 4.0
    diagnostics["routing"].mean().backward()
    assert prior.grad is not None


def test_official_route_masks_inactive_keys_and_gradients() -> None:
    attention = OfficialMiTARouteAttention(
        16,
        num_heads=2,
        router_count=4,
        top_k=4,
    )
    query = torch.randn(1, 5, 16, requires_grad=True)
    memory = torch.randn(1, 8, 16, requires_grad=True)
    key_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)

    output, diagnostics = attention(
        query,
        memory,
        key_value_mask=key_mask,
    )
    output.mean().backward()

    assert torch.isfinite(output).all()
    assert float(diagnostics["active_key_count"]) == 2.0
    assert torch.allclose(memory.grad[0, 2:], torch.zeros_like(memory.grad[0, 2:]))

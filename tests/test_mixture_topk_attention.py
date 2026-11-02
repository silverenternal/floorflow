from __future__ import annotations

import torch

from floorflow.model.family_stream_mixer import SemanticFamilyStreamMixer
from floorflow.model.mixture_topk_attention import MixtureTopKAttention


def test_mita_attention_shapes_routing_and_gradients() -> None:
    attention = MixtureTopKAttention(
        32,
        num_heads=4,
        router_count=8,
        top_k=5,
    )
    query = torch.randn(2, 16, 32, requires_grad=True)
    memory = torch.randn(2, 11, 32, requires_grad=True)
    output, diagnostics = attention(query, memory)
    assert output.shape == query.shape
    assert diagnostics["router_usage"].shape == (8,)
    assert torch.allclose(
        diagnostics["router_usage"].sum(),
        torch.tensor(1.0),
        atol=1.0e-5,
    )
    assert int(diagnostics["top_k"]) == 5
    for key in (
        "routing_entropy",
        "attention_entropy",
        "effective_expert_fraction",
    ):
        assert 0.0 <= float(diagnostics[key].detach()) <= 1.0
    output.mean().backward()
    assert query.grad is not None and memory.grad is not None


def test_mita_supports_query_priors_and_variable_expert_budgets() -> None:
    attention = MixtureTopKAttention(
        16,
        num_heads=2,
        router_count=4,
        top_k=4,
    )
    query = torch.randn(2, 8, 16)
    memory = torch.randn(2, 10, 16)
    prior = torch.randn(2, 8, 4, requires_grad=True)
    budgets = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    output, diagnostics = attention(
        query,
        memory,
        router_prior=prior,
        router_budgets=budgets,
    )
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert float(diagnostics["router_budget_min"]) == 1.0
    assert float(diagnostics["router_budget_max"]) == 4.0
    diagnostics["routing"].mean().backward()
    assert prior.grad is not None


def test_family_stream_mixer_can_use_mita() -> None:
    mixer = SemanticFamilyStreamMixer(
        16,
        attention_type="mita",
        router_count=4,
        top_k=3,
    )
    hidden = torch.randn(1, 16, 4, 4)
    condition = torch.randn(1, 16, 4, 4)
    topology = torch.randn(1, 3, 16)
    output, diagnostics = mixer(hidden, condition, topology)
    assert output.shape == hidden.shape
    assert set(mixer.last_attention_diagnostics) == set(mixer.stream_names)
    assert diagnostics["stream_gates"].shape == (1, 4)
    for key in (
        "mita_routing_entropy",
        "mita_attention_entropy",
        "mita_effective_expert_fraction",
    ):
        assert torch.isfinite(diagnostics[key])
        assert 0.0 <= float(diagnostics[key].detach()) <= 1.0


def test_family_stream_mixer_can_use_official_route_only_mita() -> None:
    mixer = SemanticFamilyStreamMixer(
        16,
        attention_type="mita",
        attention_impl="official",
        router_count=4,
        top_k=3,
    )
    output, diagnostics = mixer(
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 3, 16),
    )
    assert output.shape == (1, 16, 4, 4)
    assert mixer.attention_impl == "official"
    assert torch.isfinite(diagnostics["mita_routing_entropy"])

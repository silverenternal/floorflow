from __future__ import annotations

import torch

from floorflow.heads.algorithmic_dynamic_head import (
    DynamicKernelOutputHead,
    DynamicQueryMaskHead,
    dynamic_routing_loss,
)


def test_generated_kernel_head_has_sample_dependent_routing_and_gradients() -> None:
    head = DynamicKernelOutputHead(
        12,
        5,
        kernel_configs=((1, 1), (3, 1), (5, 2)),
        kernel_bases=3,
    )
    features = torch.randn(2, 12, 8, 8)
    output = head(features)
    assert output.shape == (2, 5, 8, 8)
    assert head.last_scale_weights is not None
    assert head.last_scale_weights_raw is not None
    assert head.last_scale_weights_raw.requires_grad
    assert head.last_scale_weights.shape == (2, 3, 5, 8, 8)
    assert torch.allclose(
        head.last_scale_weights.sum(dim=1),
        torch.ones(2, 5, 8, 8),
    )
    assert head.last_kernel_coefficients is not None
    assert head.last_kernel_coefficients.shape == (2, 3, 3)
    output.mean().backward()
    assert head.generators[1].generator[-1].weight.grad is not None
    assert head.router[-1].weight.grad is not None


def test_generated_kernels_vary_from_sample_context() -> None:
    head = DynamicKernelOutputHead(
        6,
        3,
        kernel_configs=((3, 1),),
        kernel_bases=3,
    )
    with torch.no_grad():
        head.generators[0].generator[-1].weight.normal_()
    first, _ = head.generators[0](torch.randn(1, 6, 8, 8))
    second, _ = head.generators[0](torch.randn(1, 6, 8, 8) + 3.0)
    assert not torch.allclose(first, second)


def test_dynamic_query_head_respects_budget_and_backpropagates() -> None:
    head = DynamicQueryMaskHead(
        12,
        4,
        max_queries=8,
        hidden_channels=16,
        token_grid=4,
    )
    output, diagnostics = head(torch.randn(2, 12, 8, 8), minimum_queries=2)
    assert output.shape == (2, 4, 8, 8)
    assert torch.all(diagnostics["query_budget"] >= 2.0)
    assert torch.all(diagnostics["query_budget"] <= 8.0)
    output.mean().backward()
    assert head.queries.grad is not None
    assert head.budget_head.weight.grad is not None


def test_dynamic_routing_loss_is_bounded_and_differentiable() -> None:
    weights = torch.softmax(
        torch.randn(2, 3, 4, 5, 5, requires_grad=True),
        dim=1,
    )
    loss, diagnostics = dynamic_routing_loss(weights)
    assert torch.isfinite(loss)
    loss.backward()
    assert set(diagnostics) == {
        "dead_expert_fraction",
        "routing_balance",
        "routing_entropy",
        "routing_entropy_penalty",
    }


def test_dynamic_heads_run_under_bfloat16_autocast() -> None:
    kernel_head = DynamicKernelOutputHead(8, 3, kernel_configs=((3, 1),))
    query_head = DynamicQueryMaskHead(
        8,
        3,
        max_queries=4,
        hidden_channels=16,
        token_grid=4,
    )
    features = torch.randn(2, 8, 8, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        kernel_output = kernel_head(features)
        query_output, _ = query_head(features)
    assert torch.isfinite(kernel_output).all()
    assert torch.isfinite(query_output).all()

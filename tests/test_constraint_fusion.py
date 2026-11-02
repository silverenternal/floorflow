"""Tests for conflict-aware constraint-gradient fusion."""

from __future__ import annotations

import torch

from floorflow.objectives.constraint_fusion import (
    apply_constraint_fusion,
    room_graph_edit_proxy,
)


def test_projection_removes_opposing_constraint_direction() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    base = (parameter - 1.0).square()
    constraint = (parameter + 1.0).square()
    diagnostics = apply_constraint_fusion(
        base,
        constraint,
        [parameter],
        mode="projection",
        constraint_scale=1.0,
    )
    assert diagnostics.cosine < 0.0
    assert diagnostics.projected_constraint_norm < 1.0e-6
    assert parameter.grad is not None
    assert torch.allclose(parameter.grad, torch.tensor(-2.0), atol=1.0e-6)


def test_scalar_fusion_keeps_constraint_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    base = (parameter - 1.0).square()
    constraint = (parameter + 1.0).square()
    apply_constraint_fusion(
        base,
        constraint,
        [parameter],
        mode="scalar",
        constraint_scale=0.5,
    )
    assert parameter.grad is not None
    assert torch.allclose(parameter.grad, torch.tensor(-1.0))


def test_room_graph_proxy_prefers_matching_plan_and_is_differentiable() -> None:
    target = torch.zeros(1, 3, 16, 16)
    target[:, 0, :8] = 1.0
    target[:, 1, 8:, :8] = 1.0
    target[:, 2, 8:, 8:] = 1.0
    predicted = target.clone().requires_grad_(True)
    matching, _ = room_graph_edit_proxy(predicted, target)
    changed = predicted.detach().clone()
    changed[:, 2, 8:, 8:] = 0.0
    changed[:, 0, 8:, 8:] = 1.0
    changed.requires_grad_(True)
    mismatched, _ = room_graph_edit_proxy(changed, target)
    assert matching < mismatched
    matching.backward()
    assert predicted.grad is not None

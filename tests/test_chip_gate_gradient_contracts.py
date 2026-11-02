from __future__ import annotations

import torch

from floorflow.chip_flow.global_spec import GlobalSpecConditioner
from floorflow.chip_flow.netlist_hypergraph import (
    NetlistHyperedgeConditioner,
    NetlistHypergraphContext,
)
from floorflow.chip_flow.semantic_conditioning import MaskedSemanticEncoder
from floorflow.chip_flow.structural_relations import StructuralRelationHead


def _assert_first_step_gate_gradient(gate: torch.Tensor) -> None:
    assert gate.grad is not None
    assert gate.grad.abs().sum() > 0


def test_global_spec_zero_gate_is_identity_and_gradient_reachable() -> None:
    torch.manual_seed(1)
    model = GlobalSpecConditioner(hidden=16)
    output = model(torch.randn(3, 7))
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    _assert_first_step_gate_gradient(model.gate)


def test_semantic_zero_gate_is_identity_and_gradient_reachable() -> None:
    torch.manual_seed(2)
    model = MaskedSemanticEncoder(input_dim=4, hidden=16)
    output = model(torch.randn(2, 5, 4))
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    _assert_first_step_gate_gradient(model.gate)


def test_netlist_zero_gate_is_identity_and_gradient_reachable() -> None:
    torch.manual_seed(3)
    model = NetlistHyperedgeConditioner(16, feature_dim=4)
    incidence = torch.tensor([[[True, True, False], [False, True, True]]])
    context = NetlistHypergraphContext(
        features=torch.randn(1, 2, 4),
        block_incidence=incidence,
        mask=torch.ones(1, 2, dtype=torch.bool),
    )
    output = model(
        torch.randn(1, 3, 3),
        torch.randn(1, 3, 16),
        context,
    )
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    _assert_first_step_gate_gradient(model.gate)


def test_structural_token_zero_gate_is_identity_and_gradient_reachable() -> None:
    torch.manual_seed(4)
    model = StructuralRelationHead(16, bottleneck=8)
    membership = torch.zeros(1, 4, 9, dtype=torch.bool)
    membership[0, :2, 0] = True
    mask = membership.any(dim=1)
    _, residual = model(
        torch.randn(1, 4, 16),
        torch.randn(1, 4, 3),
        torch.ones(1, 4),
        membership,
        mask,
        torch.randn(1, 9, 3),
    )
    assert torch.equal(residual, torch.zeros_like(residual))
    residual.sum().backward()
    _assert_first_step_gate_gradient(model.gate)

from __future__ import annotations

import torch

from floorflow.chip_flow.head_supervision import (
    block_supervision_masks,
    port_semantic_masks,
    structured_hierarchy_flow_loss,
)
from floorflow.chip_flow.hierarchical_flow_runtime import hierarchy_flow_loss
from floorflow.chip_flow.hierarchy_state import (
    HierarchyPrediction,
    HierarchyState,
)


def test_block_masks_exclude_only_conditioned_dimensions() -> None:
    constraints = torch.zeros(1, 4, 5)
    constraints[0, 0, 0] = 1.0
    constraints[0, 1, 1] = 1.0
    constraints[0, 2, :2] = 1.0
    center, rho = block_supervision_masks(constraints)
    assert center[0, 0].all()
    assert center[0, 1].sum() == 0
    assert center[0, 2].sum() == 0
    assert center[0, 3].all()
    assert rho[0, 0].sum() == 0
    assert rho[0, 1].sum() == 0
    assert rho[0, 3].sum() == 1


def test_port_masks_distinguish_connected_and_unconnected_pins() -> None:
    incidence = torch.zeros(1, 4, 5, dtype=torch.bool)
    incidence[0, 0, 0] = True
    incidence[0, 1, 2] = True
    port_mask = torch.tensor([[True, True, True, True, False]])
    connected, unconnected = port_semantic_masks(incidence, port_mask)
    assert connected.tolist() == [[True, False, True, False, False]]
    assert unconnected.tolist() == [[False, True, False, True, False]]


def test_structured_loss_gives_separate_semantic_terms() -> None:
    torch.manual_seed(23)
    batch, count, slots, pins = 1, 4, 3, 4
    constraints = torch.zeros(batch, count, 5)
    constraints[0, 0, 0] = 1.0
    constraints[0, 1, 1] = 1.0
    structural_mask = torch.tensor([[True, True, False]])
    incidence = torch.zeros(batch, count, pins, dtype=torch.bool)
    incidence[0, 0, 0] = True
    incidence[0, 1, 2] = True
    port_mask = torch.tensor([[True, True, True, True]])
    block = torch.randn(batch, count, 3, requires_grad=True)
    target = torch.randn_like(block)
    global_prediction = torch.randn(batch, 4, requires_grad=True)
    structural = torch.randn(batch, slots, 3, requires_grad=True)
    port = torch.randn(batch, pins, 2, requires_grad=True)
    terms = structured_hierarchy_flow_loss(
        block_prediction=block,
        block_target=target,
        global_prediction=global_prediction,
        global_target=torch.randn_like(global_prediction),
        structural_prediction=structural,
        structural_target=torch.randn_like(structural),
        port_prediction=port,
        port_target=torch.randn_like(port),
        constraints=constraints,
        structural_mask=structural_mask,
        port_incidence=incidence,
        port_mask=port_mask,
        huber_beta=0.1,
        global_weight=0.1,
        structural_weight=0.1,
        port_weight=0.1,
    )
    assert terms.total > 0
    assert terms.port_connected > 0
    assert terms.port_unconnected > 0
    assert terms.block_forced > 0
    terms.total.backward()
    assert block.grad is not None
    assert structural.grad is not None
    assert port.grad is not None


def test_runtime_dispatches_structured_hierarchy_loss() -> None:
    prediction = HierarchyPrediction(
        global_state=torch.randn(1, 4),
        structural_state=torch.randn(1, 3, 3),
        block_state=torch.randn(1, 4, 3),
        port_state=torch.randn(1, 4, 2),
    )
    target = HierarchyState(
        global_state=torch.randn(1, 4),
        structural_state=torch.randn(1, 3, 3),
        block_state=torch.randn(1, 4, 3),
        port_state=torch.randn(1, 4, 2),
    )
    loss, terms = hierarchy_flow_loss(
        prediction,
        target,
        huber_beta=0.1,
        global_weight=0.1,
        structural_weight=0.1,
        port_weight=0.1,
        constraints=torch.zeros(1, 4, 5),
        structural_mask=torch.ones(1, 3, dtype=torch.bool),
        port_incidence=torch.zeros(1, 4, 4, dtype=torch.bool),
        port_mask=torch.ones(1, 4, dtype=torch.bool),
        structured=True,
    )
    assert loss > 0
    assert "flow_block" in terms
    assert "flow_port_connected" in terms
    assert "flow_port_unconnected" in terms

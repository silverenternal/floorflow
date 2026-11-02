from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.objectives.cfm_structural import (
    CFMStructuralWeights,
    cfm_structural_loss,
    predicted_x1,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )


def test_predicted_x1_uses_flow_time() -> None:
    x_t = torch.zeros(2, 3, 4, 4)
    velocity = torch.ones_like(x_t)
    time = torch.tensor([0.0, 0.5])
    endpoint = predicted_x1(x_t, velocity, time)
    assert torch.allclose(endpoint[0], torch.ones_like(endpoint[0]))
    assert torch.allclose(endpoint[1], torch.full_like(endpoint[1], 0.5))


def test_structural_loss_is_finite_and_differentiable() -> None:
    taxonomy = _taxonomy()
    target = torch.zeros(1, taxonomy.num_target_channels, 16, 16)
    room = taxonomy.room_ids[0]
    target[:, room - 1, 2:14, 2:14] = 1.0
    footprint = torch.zeros(1, 1, 16, 16)
    footprint[:, :, 1:15, 1:15] = 1.0
    logits = torch.zeros_like(target, requires_grad=True)
    loss, components = cfm_structural_loss(
        logits,
        target,
        footprint,
        taxonomy,
        weights=CFMStructuralWeights(),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert "room_bce" in components
    assert "opening_wall" in components


def test_exact_endpoint_has_lower_structural_loss() -> None:
    taxonomy = _taxonomy()
    target = torch.zeros(1, taxonomy.num_target_channels, 16, 16)
    for class_id in taxonomy.room_ids[:2]:
        target[:, class_id - 1, 2:14, 2:14] = 1.0
    footprint = torch.ones(1, 1, 16, 16)
    weights = CFMStructuralWeights()
    wrong, _ = cfm_structural_loss(
        torch.zeros_like(target),
        target,
        footprint,
        taxonomy,
        weights=weights,
    )
    correct, _ = cfm_structural_loss(
        (target * 20.0) - 10.0,
        target,
        footprint,
        taxonomy,
        weights=weights,
    )
    assert correct < wrong

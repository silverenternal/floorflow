from __future__ import annotations

import torch

from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.hierarchy_consistency import hierarchy_cycle_loss
from floorflow.chip_flow.model import decode_boxes_batch


def _batch() -> dict:
    count = 5
    constraints = torch.zeros(2, count, 5)
    constraints[:, 0, 4] = 1
    constraints[:, 1, 4] = 2
    return {
        "areas": torch.ones(2, count),
        "boxes": torch.tensor(
            [[[index, 0.0, 1.0, 1.0] for index in range(count)]] * 2
        ),
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "source_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor([[[0.0, 1.0, 1.0]]] * 2),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0]]] * 2),
        "pins": torch.randn(2, 2, 2),
        "n_b2b": torch.tensor([1, 1]),
        "n_p2b": torch.tensor([1, 1]),
        "n_pins": torch.tensor([1, 1]),
    }


def test_hierarchy_cycle_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(19)
    model = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=1,
        typed_layers=1,
        head_dim=64,
    )
    torch.nn.init.normal_(model.backbone.final.linear.weight, std=0.02)
    batch = _batch()
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    prediction = model(target, torch.full((2,), 0.4), context)
    boxes = decode_boxes_batch(prediction.block_state, batch["areas"])
    loss, terms = hierarchy_cycle_loss(
        prediction,
        boxes,
        batch,
        context,
        huber_beta=0.1,
    )
    assert torch.isfinite(loss)
    assert set(terms) == {
        "hierarchy_cycle",
        "hierarchy_cycle_global",
        "hierarchy_cycle_structural",
        "hierarchy_cycle_port",
    }
    loss.backward()
    assert model.global_head[-1].weight.grad is not None
    assert model.backbone.final.linear.weight.grad is not None
    assert (
        model.backbone.final.linear.weight.grad.abs().sum() > 0
    )

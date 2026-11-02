from __future__ import annotations

import torch

from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.projector_residual_head import ProjectorResidualHead


def _batch() -> dict:
    count = 5
    constraints = torch.zeros(2, count, 5)
    constraints[:, 0, 4] = 1
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


def test_projector_residual_head_is_zero_initialized_and_trainable() -> None:
    head = ProjectorResidualHead(64)
    tokens = torch.randn(2, 5, 64)
    state = torch.randn(2, 5, 3)
    time = torch.randn(2, 64)
    output = head(tokens, state, time)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert head.network[-1].weight.grad is not None
    assert head.network[-1].weight.grad.abs().sum() > 0


def test_model_projector_residual_head_preserves_m27_output() -> None:
    base = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
    )
    residual = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        projector_residual_head=True,
    )
    incompatible = residual.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(
        name.startswith("projector_residual_head.")
        for name in incompatible.missing_keys
    )
    batch = _batch()
    context = residual.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    time = torch.full((2,), 0.4)
    base_prediction = base(target, time, context)
    prediction = residual(target, time, context)
    assert torch.equal(base_prediction.block_state, prediction.block_state)
    assert prediction.projector_residual is not None
    assert torch.equal(
        prediction.projector_residual,
        torch.zeros_like(prediction.projector_residual),
    )

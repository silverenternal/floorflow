from __future__ import annotations

from dataclasses import replace

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.hierarchical_flow_runtime import (
    hierarchy_source_state,
    integrate_hierarchy_step,
)


def _batch() -> dict:
    count = 5
    constraints = torch.zeros(2, count, 5)
    constraints[:, :, 4] = torch.tensor([[1, 0, 2, 0, 4]] * 2)
    constraints[:, :2, 2] = 1
    constraints[0, :2, 3] = 1
    constraints[1, 2:4, 3] = 2
    boxes = torch.tensor(
        [[[index * 1.2, 0.0, 1.0, 1.0] for index in range(count)]] * 2
    )
    return {
        "areas": torch.ones(2, count),
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "source_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]],
                [[0.0, 2.0, 1.0], [2.0, 4.0, 0.5]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 0.5]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.2, 0.4], [0.8, 0.6], [0.0, 0.0]],
                [[0.1, 0.5], [0.7, 0.9], [0.0, 0.0]],
            ]
        ),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }


def _model(*, self_conditioning: bool) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=1,
        typed_layers=1,
        head_dim=64,
        self_conditioning=self_conditioning,
    )


def test_first_self_conditioned_call_matches_base_and_history_is_trainable() -> None:
    torch.manual_seed(31)
    base = _model(self_conditioning=False).eval()
    candidate = _model(self_conditioning=True).eval()
    incompatible = candidate.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(
        name.startswith("history_to_latent.")
        for name in incompatible.missing_keys
    )
    final_weight = torch.randn_like(base.backbone.final.linear.weight) * 0.02
    base.backbone.final.linear.weight.data.copy_(final_weight)
    candidate.backbone.final.linear.weight.data.copy_(final_weight)
    batch = _batch()
    context = candidate.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    time = torch.full((2,), 0.4)
    baseline = base(target, time, context).block_state
    first = candidate(target, time, context).block_state
    assert torch.equal(first, baseline)

    assert candidate.history_to_latent is not None
    candidate.history_to_latent[-1].weight.data.normal_(std=0.02)
    previous = target.block_state + 0.1
    with_history = candidate(
        replace(target, previous_block_state=previous),
        time,
        context,
    )
    assert not torch.equal(with_history.block_state, first)
    with_history.block_state.sum().backward()
    assert candidate.history_to_latent[-1].weight.grad is not None
    assert candidate.history_to_latent[-1].weight.grad.abs().sum() > 0


def test_integrate_step_records_previous_x0_without_extra_call() -> None:
    torch.manual_seed(37)
    model = _model(self_conditioning=True).eval()
    batch = _batch()
    context = model.prepare_context(batch)
    source = hierarchy_source_state(
        batch["source_latent"],
        context,
        torch.Generator().manual_seed(37),
    )
    time = torch.full((2,), 0.5)
    expected = model(source, time, context).block_state
    with torch.no_grad():
        advanced, _ = integrate_hierarchy_step(
            model,
            source,
            time,
            context,
            batch,
            step_size=0.25,
            projector=False,
            projector_iterations=0,
            projector_boundary_axes=False,
            projector_boundary_lock_corners=True,
        )
    assert advanced.previous_block_state is not None
    assert torch.equal(advanced.previous_block_state, expected)


def test_self_conditioning_checkpoint_rebuilds_adapter() -> None:
    model = _model(self_conditioning=True)
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 1,
            "head_dim": 64,
            "typed_layers": 1,
            "self_conditioning": True,
        },
        "ema": model.state_dict(),
        "model": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.self_conditioning
    assert restored.history_to_latent is not None

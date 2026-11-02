from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.ordered_group_staging import OrderedGroupStaging


def _batch() -> dict:
    count = 8
    constraints = torch.zeros(2, count, 5)
    constraints[:, :, 3] = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    constraints[:, :2, 4] = 1
    return {
        "areas": torch.ones(2, count),
        "boxes": torch.stack(
            [
                torch.stack(
                    [
                        torch.arange(count) * 1.2,
                        torch.zeros(count),
                        torch.ones(count),
                        torch.ones(count),
                    ],
                    dim=-1,
                )
            ]
            * 2
        ),
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 3.0, 0.5]],
                [[0.0, 2.0, 1.0], [2.0, 4.0, 1.0], [4.0, 6.0, 0.5]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 3.0, 0.5]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 1.0], [2.0, 6.0, 0.5]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.2, 0.4], [0.8, 0.6], [0.5, 0.5]],
                [[0.1, 0.5], [0.7, 0.9], [0.4, 0.3]],
            ]
        ),
        "n_b2b": torch.tensor([3, 3]),
        "n_p2b": torch.tensor([3, 3]),
        "n_pins": torch.tensor([3, 3]),
    }


def _model(order: bool) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        order_conditioning=order,
    )


def test_ordered_group_staging_is_zero_gated() -> None:
    torch.manual_seed(3)
    module = OrderedGroupStaging(64)
    tokens = torch.randn(2, 8, 64)
    state = torch.randn(2, 8, 3)
    clusters = torch.randn(2, 4, 64)
    ids = torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4]] * 2)
    mask = torch.stack(
        [ids.eq(group) for group in range(1, 5)],
        dim=-1,
    ).any(dim=1)
    residual, diagnostics = module(
        tokens,
        state,
        clusters,
        ids,
        mask,
        torch.tensor([0.25, 0.75]),
    )
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.isfinite(diagnostics["order_group_phase_gate"]).all()
    module.gate.data.fill_(1.0)
    active, _ = module(
        tokens,
        state,
        clusters,
        ids,
        mask,
        torch.tensor([0.25, 0.75]),
    )
    assert active.abs().sum() > 0
    active.square().mean().backward()
    assert module.fuse[-1].weight.grad is not None
    assert module.fuse[-1].weight.grad.abs().sum() > 0


def test_ordered_group_staging_is_permutation_equivariant() -> None:
    torch.manual_seed(5)
    module = OrderedGroupStaging(64)
    module.gate.data.fill_(1.0)
    tokens = torch.randn(1, 8, 64)
    state = torch.randn(1, 8, 3)
    clusters = torch.randn(1, 4, 64)
    ids = torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4]])
    mask = torch.stack(
        [ids.eq(group) for group in range(1, 5)],
        dim=-1,
    ).any(dim=1)
    time = torch.tensor([0.4])
    original, _ = module(tokens, state, clusters, ids, mask, time)
    permutation = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
    reordered, _ = module(
        tokens[:, permutation],
        state[:, permutation],
        clusters,
        ids[:, permutation],
        mask,
        time,
    )
    torch.testing.assert_close(
        original[:, permutation],
        reordered,
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_order_conditioning_zero_gate_preserves_base_prediction() -> None:
    torch.manual_seed(7)
    base = _model(False)
    torch.manual_seed(7)
    staged = _model(True)
    staged.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    staged_context = staged.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)
    assert torch.equal(
        base(target, time, base_context).block_state,
        staged(target, time, staged_context).block_state,
    )
    assert staged.order_staging is not None


def test_order_conditioning_checkpoint_round_trip() -> None:
    model = _model(True)
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 2,
            "head_dim": 64,
            "typed_layers": 1,
            "order_conditioning": True,
        },
        "model": model.state_dict(),
        "ema": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.order_staging is not None


def test_order_conditioning_parameter_budget_is_under_one_percent() -> None:
    module = OrderedGroupStaging(192)
    base_parameters = 5_836_447
    added = sum(parameter.numel() for parameter in module.parameters())
    assert added < base_parameters * 0.01

from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.complexity_slot_adapter import ComplexitySlotAdapter
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)


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


def _model(complexity_slots: bool) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        complexity_slot_conditioning=complexity_slots,
    )


def test_complexity_slots_are_zero_gated_and_allocate_capacity() -> None:
    torch.manual_seed(3)
    module = ComplexitySlotAdapter(64)
    tokens = torch.randn(2, 12, 64)
    state = torch.randn(2, 12, 3)
    complexity = torch.tensor([0.05, 0.95])
    time = torch.tensor([0.25, 0.75])
    residual, diagnostics = module(tokens, state, complexity, time)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert diagnostics["complexity_slot_active_count"][1] >= (
        diagnostics["complexity_slot_active_count"][0]
    )
    module.gate.data.fill_(1.0)
    active, _ = module(tokens, state, complexity, time)
    assert active[1].abs().mean() > active[0].abs().mean()
    active.square().mean().backward()
    assert module.reduce[-1].weight.grad is not None
    assert module.reduce[-1].weight.grad.abs().sum() > 0
    assert module.router.budget_head.weight.grad is not None
    assert module.router.budget_head.weight.grad.abs().sum() > 0


def test_complexity_slots_preserve_block_permutation_equivariance() -> None:
    torch.manual_seed(5)
    module = ComplexitySlotAdapter(64)
    module.gate.data.fill_(1.0)
    tokens = torch.randn(1, 8, 64)
    state = torch.randn(1, 8, 3)
    complexity = torch.tensor([0.8])
    time = torch.tensor([0.4])
    original, _ = module(tokens, state, complexity, time)
    permutation = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
    reordered, _ = module(
        tokens[:, permutation],
        state[:, permutation],
        complexity,
        time,
    )
    torch.testing.assert_close(
        original[:, permutation],
        reordered,
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_complexity_slots_zero_gate_preserves_base_prediction() -> None:
    torch.manual_seed(7)
    base = _model(False)
    torch.manual_seed(7)
    coupled = _model(True)
    coupled.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    coupled_context = coupled.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)
    torch.testing.assert_close(
        base(target, time, base_context).block_state,
        coupled(target, time, coupled_context).block_state,
    )
    assert coupled.complexity_slot is not None


def test_complexity_slots_checkpoint_round_trip() -> None:
    model = _model(True)
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 2,
            "head_dim": 64,
            "typed_layers": 1,
            "complexity_slot_conditioning": True,
        },
        "model": model.state_dict(),
        "ema": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.complexity_slot is not None


def test_complexity_slots_parameter_budget_is_under_two_percent() -> None:
    module = ComplexitySlotAdapter(192)
    base_parameters = 5_836_447
    added = sum(parameter.numel() for parameter in module.parameters())
    assert added < base_parameters * 0.02

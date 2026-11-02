from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.coarse_to_fine_coupler import CoarseToFineCoupler
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


def _model(coarse_to_fine: bool) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        coarse_to_fine_conditioning=coarse_to_fine,
    )


def test_coarse_to_fine_is_zero_gated_and_trainable() -> None:
    torch.manual_seed(3)
    module = CoarseToFineCoupler(64)
    batch, count = 2, 8
    membership = torch.zeros(batch, count, 9, dtype=torch.bool)
    membership[:, :2, 0] = True
    membership[:, 2:4, 1] = True
    membership[:, 4:6, 2] = True
    membership[:, 6:, 3] = True
    arguments = (
        torch.randn(batch, count, 64),
        torch.randn(batch, count, 3),
        membership,
        torch.tensor([[True, True, True, True, False, False, False, False, False]] * batch),
        torch.randn(batch, 4),
        torch.randn(batch, 9, 3),
        torch.tensor([0.25, 0.75]),
    )
    residual, diagnostics = module(*arguments)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert diagnostics["coarse_to_fine_active_fraction"] == 1.0
    module.gate.data.fill_(1.0)
    active, _ = module(*arguments)
    assert active.abs().sum() > 0
    active.square().mean().backward()
    assert module.reduce[-1].weight.grad is not None
    assert module.reduce[-1].weight.grad.abs().sum() > 0


def test_coarse_to_fine_is_block_permutation_equivariant() -> None:
    torch.manual_seed(5)
    module = CoarseToFineCoupler(64)
    module.gate.data.fill_(1.0)
    tokens = torch.randn(1, 8, 64)
    state = torch.randn(1, 8, 3)
    membership = torch.zeros(1, 8, 9, dtype=torch.bool)
    membership[0, :, 0] = True
    mask = torch.zeros(1, 9, dtype=torch.bool)
    mask[0, 0] = True
    global_prediction = torch.randn(1, 4)
    structural_prediction = torch.randn(1, 9, 3)
    time = torch.tensor([0.4])
    original, _ = module(
        tokens,
        state,
        membership,
        mask,
        global_prediction,
        structural_prediction,
        time,
    )
    permutation = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
    reordered, _ = module(
        tokens[:, permutation],
        state[:, permutation],
        membership[:, permutation],
        mask,
        global_prediction,
        structural_prediction,
        time,
    )
    torch.testing.assert_close(
        original[:, permutation],
        reordered,
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_coarse_to_fine_zero_gate_preserves_base_prediction() -> None:
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
    assert coupled.coarse_to_fine is not None


def test_coarse_to_fine_checkpoint_round_trip() -> None:
    model = _model(True)
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 2,
            "head_dim": 64,
            "typed_layers": 1,
            "coarse_to_fine_conditioning": True,
        },
        "model": model.state_dict(),
        "ema": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.coarse_to_fine is not None


def test_coarse_to_fine_parameter_budget_is_under_one_percent() -> None:
    module = CoarseToFineCoupler(192)
    base_parameters = 5_836_447
    added = sum(parameter.numel() for parameter in module.parameters())
    assert added < base_parameters * 0.01

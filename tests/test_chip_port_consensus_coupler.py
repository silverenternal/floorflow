from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.port_consensus_coupler import PortConsensusCoupler


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


def _model(port_consensus: bool) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        port_consensus_conditioning=port_consensus,
    )


def test_port_consensus_is_zero_gated_and_trainable() -> None:
    torch.manual_seed(3)
    module = PortConsensusCoupler(64)
    incidence = torch.zeros(2, 8, 4, dtype=torch.bool)
    incidence[:, 0, 0] = True
    incidence[:, 1, 0] = True
    incidence[:, 2, 1] = True
    arguments = (
        torch.randn(2, 8, 64),
        torch.randn(2, 8, 3),
        torch.randn(2, 4, 2),
        incidence,
        torch.tensor(
            [[True, True, True, False], [True, True, True, False]]
        ),
        torch.tensor([0.25, 0.75]),
    )
    residual, diagnostics = module(*arguments)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert diagnostics["port_consensus_active_fraction"] == 0.375
    module.gate.data.fill_(1.0)
    active, _ = module(*arguments)
    assert active.abs().sum() > 0
    active.square().mean().backward()
    assert module.reduce[-1].weight.grad is not None
    assert module.reduce[-1].weight.grad.abs().sum() > 0


def test_port_consensus_is_block_permutation_equivariant() -> None:
    torch.manual_seed(5)
    module = PortConsensusCoupler(64)
    module.gate.data.fill_(1.0)
    tokens = torch.randn(1, 8, 64)
    state = torch.randn(1, 8, 3)
    ports = torch.randn(1, 4, 2)
    incidence = torch.zeros(1, 8, 4, dtype=torch.bool)
    incidence[0, :, 0] = True
    mask = torch.tensor([[True, False, False, False]])
    time = torch.tensor([0.4])
    original, _ = module(tokens, state, ports, incidence, mask, time)
    permutation = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
    reordered, _ = module(
        tokens[:, permutation],
        state[:, permutation],
        ports,
        incidence[:, permutation],
        mask,
        time,
    )
    torch.testing.assert_close(
        original[:, permutation],
        reordered,
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_port_consensus_zero_gate_preserves_base_prediction() -> None:
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
    assert coupled.port_consensus is not None


def test_port_consensus_checkpoint_round_trip() -> None:
    model = _model(True)
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 2,
            "head_dim": 64,
            "typed_layers": 1,
            "port_consensus_conditioning": True,
        },
        "model": model.state_dict(),
        "ema": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.port_consensus is not None


def test_old_port_consensus_checkpoint_accepts_geometry_scale() -> None:
    model = _model(True)
    legacy = {
        key: value
        for key, value in model.state_dict().items()
        if key != "port_consensus.geometry_scale"
    }
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 2,
            "head_dim": 64,
            "typed_layers": 1,
            "port_consensus_conditioning": True,
        },
        "model": legacy,
    }

    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)

    assert restored.port_consensus.geometry_scale.item() == 0.0


def test_port_consensus_parameter_budget_is_under_one_percent() -> None:
    module = PortConsensusCoupler(192)
    base_parameters = 5_836_447
    added = sum(parameter.numel() for parameter in module.parameters())
    assert added < base_parameters * 0.01


def test_port_consensus_rho_residual_respects_trust_radius() -> None:
    module = PortConsensusCoupler(16, rho_trust_radius=0.05)
    module.gate.data.fill_(1.0)
    torch.nn.init.normal_(module.reduce[-1].weight, std=1.0)
    tokens = torch.randn(1, 2, 16)
    state = torch.randn(1, 2, 3)
    ports = torch.randn(1, 1, 2)
    incidence = torch.ones(1, 2, 1, dtype=torch.bool)
    mask = torch.ones(1, 1, dtype=torch.bool)
    residual, _ = module(tokens, state, ports, incidence, mask, torch.tensor([0.5]))
    assert residual[..., 2].abs().max() <= 0.05 + 1.0e-6
    assert residual[..., :2].abs().max() <= 0.25 + 1.0e-6


def test_port_consensus_uses_pin_plus_offset_geometry() -> None:
    module = PortConsensusCoupler(
        16,
        rho_trust_radius=0.05,
        position_trust_radius=1.0,
    )
    module.geometry_scale.data.fill_(1.0)
    tokens = torch.zeros(1, 2, 16)
    state = torch.zeros(1, 2, 3)
    ports = torch.tensor([[[0.2, -0.1]]])
    incidence = torch.tensor([[[True], [False]]])
    port_mask = torch.tensor([[True]])
    pin_positions = torch.tensor([[[0.5, 0.4]]])
    pin_mask = torch.tensor([[True]])

    residual, diagnostics = module(
        tokens,
        state,
        ports,
        incidence,
        port_mask,
        torch.tensor([0.5]),
        pin_positions=pin_positions,
        pin_mask=pin_mask,
    )

    expected = torch.tanh(torch.tensor(1.0)) * torch.tensor([0.7, 0.3])
    torch.testing.assert_close(residual[0, 0, :2], expected)
    assert torch.count_nonzero(residual[0, 1]) == 0
    assert diagnostics["port_consensus_geometry_scale"].item() == 1.0

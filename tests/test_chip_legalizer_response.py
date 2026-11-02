from __future__ import annotations

from types import SimpleNamespace

import torch

from floorflow.chip_flow.hierarchy_adapters import legalizer_action_radius
from floorflow.chip_flow.legalizer_response import (
    LegalizerResponseHead,
    apply_legalizer_displacement,
    legalizer_response_from_scale_pack,
    legalizer_response_loss,
    legalizer_response_targets,
)


def test_legalizer_response_head_is_zero_initialized_and_trainable() -> None:
    module = LegalizerResponseHead(32, bottleneck=8)
    tokens = torch.randn(2, 4, 32)
    state = torch.randn(2, 4, 3)
    prediction = torch.randn(2, 4, 3)
    time_token = torch.randn(2, 32)
    displacement, rung = module(tokens, state, prediction, time_token)
    assert displacement.shape == (2, 4, 2)
    assert rung.shape == (2, 5)
    assert torch.equal(displacement, torch.zeros_like(displacement))
    assert torch.equal(rung, torch.zeros_like(rung))


def test_legalizer_response_loss_receives_gradient() -> None:
    module = LegalizerResponseHead(32, bottleneck=8)
    tokens = torch.randn(2, 4, 32)
    state = torch.randn(2, 4, 3)
    prediction = torch.randn(2, 4, 3)
    time_token = torch.randn(2, 32)
    displacement, rung = module(tokens, state, prediction, time_token)
    batch = {
        "legalizer_displacement": torch.randn(2, 4, 2),
        "legalizer_rung": torch.tensor([2, 3]),
    }
    terms = legalizer_response_loss(displacement, rung, batch)
    assert torch.isfinite(terms.total)
    terms.total.backward()
    assert module.displacement.weight.grad is not None
    assert module.rung.weight.grad is not None
    assert module.displacement.weight.grad.abs().sum() > 0
    assert module.rung.weight.grad.abs().sum() > 0


def test_legalizer_response_ignores_preplaced_and_invalid_blocks() -> None:
    prediction = torch.zeros(1, 2, 2)
    target = torch.tensor([[[10.0, -10.0], [0.0, 0.0]]])
    batch = {
        "legalizer_displacement": target,
        "legalizer_rung": torch.tensor([2]),
        "legalizer_mask": torch.tensor([[False, True]]),
        "constraints": torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]]
        ),
    }

    terms = legalizer_response_loss(
        prediction,
        torch.zeros(1, 5),
        batch,
        rung_weight=0.0,
    )

    assert terms.displacement == 0


def test_legalizer_action_preserves_masked_centers() -> None:
    prediction = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]])
    displacement = torch.ones(1, 2, 2)

    updated = apply_legalizer_displacement(
        prediction,
        displacement,
        center_mask=torch.tensor([[True, False]]),
        max_displacement=1.0,
    )

    torch.testing.assert_close(updated[0, 1], prediction[0, 1])
    assert not torch.equal(updated[0, 0], prediction[0, 0])


def test_legalizer_action_respects_per_block_radius() -> None:
    prediction = torch.zeros(1, 2, 3)
    displacement = torch.ones(1, 2, 2)

    updated = apply_legalizer_displacement(
        prediction,
        displacement,
        max_displacement=1.0,
        per_block_radius=torch.tensor([[0.02, 0.4]]),
    )

    torch.testing.assert_close(
        updated[0, :, :2].norm(dim=-1),
        torch.tensor([0.02, 0.4]),
    )


def test_legalizer_radius_is_bounded_by_nearest_pin_distance() -> None:
    prediction = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]]
    )
    context = SimpleNamespace(
        pin_positions=torch.tensor([[[0.2, 0.1]]]),
        typed=SimpleNamespace(
            port_block_incidence=torch.tensor([[[True], [False]]]),
            port_mask=torch.tensor([[True]]),
        ),
        edge_state=None,
    )

    radius = legalizer_action_radius(
        prediction,
        context,
        max_displacement=0.1,
    )

    torch.testing.assert_close(radius, torch.tensor([[0.03, 0.1]]))


def test_legalizer_radius_is_bounded_by_nearest_net_distance() -> None:
    prediction = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]]
    )
    context = SimpleNamespace(
        pin_positions=None,
        typed=SimpleNamespace(
            port_block_incidence=torch.zeros(1, 2, 1, dtype=torch.bool),
            port_mask=torch.zeros(1, 1, dtype=torch.bool),
        ),
        edge_state=SimpleNamespace(
            cache={
                "b2b_source": torch.tensor([0]),
                "b2b_target": torch.tensor([1]),
            }
        ),
    )

    radius = legalizer_action_radius(
        prediction,
        context,
        max_displacement=1.0,
    )

    torch.testing.assert_close(radius, torch.tensor([[0.1, 0.1]]))


def test_legalizer_response_targets_use_existing_telemetry() -> None:
    areas = torch.tensor([[1.0, 1.0]])
    before = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]]])
    after = before.clone()
    after[..., :2] += torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    targets = legalizer_response_targets(
        before,
        after,
        torch.tensor([3]),
        areas,
    )
    assert targets["legalizer_displacement"].shape == (1, 2, 2)
    assert targets["legalizer_rung"].tolist() == [3]
    assert torch.isfinite(targets["legalizer_displacement"]).all()
    telemetry = legalizer_response_from_scale_pack(
        before,
        after,
        areas,
        {"rung": 3},
    )
    torch.testing.assert_close(
        telemetry["legalizer_displacement"],
        targets["legalizer_displacement"],
    )

from __future__ import annotations

import pytest
import torch

from floorflow.objectives.gradient_balancer import GradientNormBalancer


def test_gradient_balancer_reduces_dominant_term() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    losses = {
        "flow": parameter[0].square(),
        "semantic": 10.0 * parameter[1].square(),
    }
    balancer = GradientNormBalancer(
        {"flow": 1.0, "semantic": 1.0},
        update_interval=1,
        momentum=0.0,
    )
    balancer.maybe_update(losses, [parameter], step=0)
    assert balancer.scales["semantic"] == 1.0
    balancer.maybe_update(losses, [parameter], step=1)
    assert balancer.scales["flow"] == 1.0
    assert balancer.scales["semantic"] < 0.2
    diagnostics = balancer.diagnostics()
    assert diagnostics["last_update_step"] == 1
    assert diagnostics["last_active_parameters"]["flow"] == 1
    assert diagnostics["last_active_parameters"]["semantic"] == 1
    balanced = balancer.weighted_loss(losses)
    balanced.backward()
    assert parameter.grad is not None


def test_gradient_balancer_round_trip() -> None:
    balancer = GradientNormBalancer({"flow": 1.0, "semantic": 0.5})
    balancer.norm_ema = {"flow": 1.0, "semantic": 2.0}
    balancer.scales = {"flow": 1.0, "semantic": 0.5}
    balancer.last_norms = {"flow": 1.0, "semantic": 2.0}
    balancer.last_active_parameters = {"flow": 2, "semantic": 3}
    balancer.last_update_step = 7
    restored = GradientNormBalancer({"flow": 1.0, "semantic": 0.5})
    restored.load_state_dict(balancer.state_dict())
    assert restored.norm_ema == balancer.norm_ema
    assert restored.scales == balancer.scales
    assert restored.diagnostics() == balancer.diagnostics()


def test_gradient_balancer_reset_scales_when_reference_changes() -> None:
    old = GradientNormBalancer(
        {"ce": 1.0, "instance_room": 1.0},
        reference="ce",
    )
    old.norm_ema = {"ce": 0.1, "instance_room": 5.0}
    old.scales = {"ce": 1.0, "instance_room": 0.2}
    restored = GradientNormBalancer(
        {"ce": 1.0, "instance_room": 1.0},
        reference="instance_room",
    )
    restored.load_state_dict(old.state_dict())
    assert restored.reference == "instance_room"
    assert restored.norm_ema == {}
    assert restored.scales == {"ce": 1.0, "instance_room": 1.0}
    assert restored.last_update_step == -1


def test_gradient_balancer_accepts_new_defaulted_terms() -> None:
    old = GradientNormBalancer({"flow": 1.0, "semantic": 1.0})
    old.norm_ema = {"flow": 1.0, "semantic": 2.0}
    old.scales = {"flow": 1.0, "semantic": 0.5}
    restored = GradientNormBalancer(
        {"flow": 1.0, "semantic": 1.0, "new_term": 0.2}
    )
    restored.load_state_dict(old.state_dict())
    assert restored.scales["semantic"] == 0.5
    assert restored.scales["new_term"] == 1.0
    assert "new_term" not in restored.norm_ema


def test_gradient_balancer_norm_skips_unused_parameters() -> None:
    used = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    unused = torch.nn.Parameter(torch.tensor([1.0]))
    balancer = GradientNormBalancer({"flow": 1.0})
    norm = balancer._gradient_norm(used.square().sum(), [used, unused])
    assert norm.item() == pytest.approx(10.0)


def test_zero_weight_constant_terms_are_ignored() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    losses = {
        "flow": parameter.square(),
        "disabled": torch.zeros(()),
    }
    balancer = GradientNormBalancer(
        {"flow": 1.0, "disabled": 0.0},
        update_interval=1,
    )
    balancer.maybe_update(losses, [parameter], step=1)
    assert "disabled" not in balancer.diagnostics()["last_norms"]
    assert torch.isfinite(balancer.weighted_loss(losses))


def test_fractional_balance_exponent_softens_scale() -> None:
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    losses = {"flow": first.square(), "semantic": second.square()}
    balancer = GradientNormBalancer(
        {"flow": 1.0, "semantic": 1.0},
        update_interval=1,
        momentum=0.0,
        exponent=0.5,
    )
    balancer.maybe_update(losses, [first, second], step=1)
    assert balancer.scales["semantic"] == pytest.approx(0.5**0.5)


def test_gradient_balancer_ignores_frozen_parameters() -> None:
    trainable = torch.nn.Parameter(torch.tensor(2.0))
    frozen = torch.nn.Parameter(torch.tensor(3.0), requires_grad=False)
    balancer = GradientNormBalancer(
        {"flow": 1.0},
        update_interval=1,
    )
    loss = trainable.square() + frozen.square()
    balancer.maybe_update(
        losses={"flow": loss},
        parameters=[trainable, frozen],
        step=1,
    )
    assert balancer.diagnostics()["last_active_parameters"]["flow"] == 1

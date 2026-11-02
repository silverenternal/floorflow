from __future__ import annotations

import pytest
import torch
from torch import nn

from floorflow.training.cfm_convergence import (
    CFMConvergenceConfig,
    effective_updates,
    make_joint_ema,
    make_joint_optimizer,
    make_joint_scheduler,
    make_style_optimizer,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)


class _StyledModel(_TinyModel):
    def __init__(self) -> None:
        super().__init__()
        self.hub = nn.Module()
        self.hub.pixel = nn.Module()
        self.hub.pixel.style_adapter = nn.Linear(2, 2)


def test_effective_updates_counts_final_partial_group() -> None:
    assert effective_updates(10, 4) == 3
    assert effective_updates(0, 4) == 0
    assert effective_updates(8, 1) == 8


def test_joint_optimizer_scheduler_and_ema_share_update_clock() -> None:
    model = _TinyModel()
    config = CFMConvergenceConfig(
        base_lr=1.0e-5,
        head_lr=1.0e-4,
        warmup_fraction=0.2,
        ema_decay=0.5,
        fused_adamw=False,
    )
    optimizer = make_joint_optimizer(model, config)
    scheduler = make_joint_scheduler(optimizer, config, total_updates=10)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1.0e-5, 1.0e-4]
    )
    scheduler.step()
    assert optimizer.param_groups[1]["lr"] < config.head_lr
    ema = make_joint_ema(model, config)
    with torch.no_grad():
        model.head.weight.add_(1.0)
    ema.update(model)
    assert not torch.equal(ema.module.head.weight, model.head.weight)


def test_convergence_config_rejects_invalid_accumulation() -> None:
    with pytest.raises(ValueError, match="accumulation"):
        CFMConvergenceConfig(gradient_accumulation_steps=0)


def test_style_parameters_use_a_separate_optimizer() -> None:
    model = _StyledModel()
    config = CFMConvergenceConfig(fused_adamw=False)
    optimizer = make_joint_optimizer(model, config)
    style_optimizer = make_style_optimizer(model, lr=1.0e-2, weight_decay=0.0)
    main_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    style_ids = {
        id(parameter)
        for group in style_optimizer.param_groups
        for parameter in group["params"]
    }
    assert main_ids.isdisjoint(style_ids)
    assert style_ids == {
        id(parameter)
        for parameter in model.hub.pixel.style_adapter.parameters()
    }

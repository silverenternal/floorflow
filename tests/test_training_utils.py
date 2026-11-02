from __future__ import annotations

import torch

from floorflow.training.training import ModelEMA, WarmupCosineScheduler, auxiliary_weight


def test_warmup_cosine_reaches_minimum() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=100,
        base_lr=1.0,
        warmup_fraction=0.1,
        min_lr_ratio=0.2,
    )
    assert scheduler.lr_at(1) < scheduler.lr_at(10)
    assert abs(scheduler.lr_at(100) - 0.2) < 1.0e-6


def test_schedule_can_be_applied_before_first_optimizer_step() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=10,
        base_lr=1.0,
        warmup_fraction=0.2,
        min_lr_ratio=0.1,
    )
    assert scheduler.apply(1) == 0.5
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_model_ema_moves_toward_source() -> None:
    model = torch.nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.5)
    before = ema.module.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    assert not torch.allclose(before, ema.module.weight)


def test_auxiliary_weight_ramp() -> None:
    assert auxiliary_weight(0.8, 0, 4) == 0.2
    assert auxiliary_weight(0.8, 1, 4) == 0.4
    assert auxiliary_weight(0.8, 5, 4) == 0.8


def test_scheduler_preserves_parameter_group_ratios() -> None:
    left = torch.nn.Parameter(torch.tensor(1.0))
    right = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD(
        [{"params": [left], "lr": 0.2}, {"params": [right], "lr": 1.0}],
        lr=1.0,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=10,
        base_lr=1.0,
        warmup_fraction=0.0,
    )
    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - scheduler.base_lr * 0.2) < 1.0e-8
    assert abs(optimizer.param_groups[1]["lr"] - scheduler.base_lr) < 1.0e-8

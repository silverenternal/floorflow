"""Optimization controls for fast, resumable conditional-flow training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from floorflow.training.training import ModelEMA, WarmupCosineScheduler


@dataclass(frozen=True)
class CFMConvergenceConfig:
    """Validated optimizer and schedule settings shared by CFM trainers."""

    base_lr: float = 2.0e-6
    head_lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    gradient_accumulation_steps: int = 1
    warmup_fraction: float = 0.04
    min_lr_ratio: float = 0.10
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    fused_adamw: bool = True

    def __post_init__(self) -> None:
        if self.base_lr <= 0.0 or self.head_lr <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.grad_clip < 0.0:
            raise ValueError("grad_clip must be non-negative")

    def manifest(self) -> dict[str, float | int | bool]:
        return asdict(self)


def effective_updates(batches: int, accumulation_steps: int) -> int:
    """Return optimizer updates, including the final partial accumulation group."""

    batches = max(0, int(batches))
    accumulation_steps = max(1, int(accumulation_steps))
    return math.ceil(batches / accumulation_steps)


def is_style_parameter(name: str) -> bool:
    """Identify every learnable style prompt/residual, including new modules."""

    return "style" in name.lower()


def make_joint_optimizer(
    model: torch.nn.Module,
    config: CFMConvergenceConfig,
) -> torch.optim.Optimizer:
    """Create grouped AdamW and use fused CUDA kernels when available."""

    base_parameters = list(model.base.parameters())
    base_ids = {id(parameter) for parameter in base_parameters}
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if (
            id(parameter) not in base_ids
            and parameter.requires_grad
            and not is_style_parameter(name)
        )
    ]
    groups = []
    if base_parameters:
        groups.append({"params": base_parameters, "lr": config.base_lr})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": config.head_lr})
    if not groups:
        raise ValueError("model has no trainable parameters")
    fused = bool(config.fused_adamw and torch.cuda.is_available())
    return torch.optim.AdamW(
        groups,
        weight_decay=config.weight_decay,
        fused=fused,
    )


def make_style_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer | None:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if is_style_parameter(name) and parameter.requires_grad
    ]
    if not parameters:
        return None
    return torch.optim.AdamW(
        parameters,
        lr=float(lr),
        weight_decay=float(weight_decay),
    )


def make_joint_scheduler(
    optimizer: torch.optim.Optimizer,
    config: CFMConvergenceConfig,
    total_updates: int,
) -> WarmupCosineScheduler:
    """Schedule all parameter groups in one effective-update clock."""

    return WarmupCosineScheduler(
        optimizer,
        total_steps=max(1, int(total_updates)),
        base_lr=config.head_lr,
        warmup_fraction=config.warmup_fraction,
        min_lr_ratio=config.min_lr_ratio,
    )


def make_joint_ema(
    model: torch.nn.Module,
    config: CFMConvergenceConfig,
) -> ModelEMA:
    return ModelEMA(model, decay=config.ema_decay)


__all__ = [
    "CFMConvergenceConfig",
    "effective_updates",
    "is_style_parameter",
    "make_joint_ema",
    "make_joint_optimizer",
    "make_joint_scheduler",
    "make_style_optimizer",
]

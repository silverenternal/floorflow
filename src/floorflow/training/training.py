"""Reusable training-loop utilities adapted from multiflow training practice."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        base_lr: float,
        warmup_fraction: float = 0.08,
        min_lr_ratio: float = 0.15,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.base_lr = float(base_lr)
        self.group_lr_ratios = [
            float(group.get("lr", base_lr)) / max(float(base_lr), 1.0e-12)
            for group in optimizer.param_groups
        ]
        self.warmup_steps = max(
            0,
            round(self.total_steps * max(0.0, float(warmup_fraction))),
        )
        self.min_lr_ratio = max(0.0, min(1.0, float(min_lr_ratio)))
        self.step_index = 0

    def lr_at(self, step_index: int) -> float:
        step_index = max(1, int(step_index))
        if self.warmup_steps > 0 and step_index <= self.warmup_steps:
            ratio = step_index / self.warmup_steps
        else:
            progress = (step_index - self.warmup_steps - 1) / max(1, self.total_steps - self.warmup_steps - 1)
            progress = max(0.0, progress)
            ratio = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * min(1.0, progress))
            )
        return self.base_lr * ratio

    def apply(self, step_index: int | None = None) -> float:
        """Apply the schedule before an optimizer step or after resume."""

        if step_index is None:
            step_index = max(1, self.step_index + 1)
        else:
            step_index = max(1, int(step_index))
        learning_rate = self.lr_at(step_index)
        for group, ratio in zip(self.optimizer.param_groups, self.group_lr_ratios):
            group["lr"] = learning_rate * ratio
        return learning_rate

    def step(self) -> float:
        self.step_index += 1
        return self.apply(self.step_index)

    def state_dict(self) -> dict[str, float | int]:
        return {
            "total_steps": self.total_steps,
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "step_index": self.step_index,
        }

    def load_state_dict(self, state: dict) -> None:
        self.total_steps = int(state.get("total_steps", self.total_steps))
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.min_lr_ratio = float(state.get("min_lr_ratio", self.min_lr_ratio))
        self.step_index = int(state.get("step_index", self.step_index))


class ModelEMA:
    """EMA shadow weights for evaluation checkpoints."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()
        for name, value in self.module.state_dict().items():
            source = model_state[name]
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(source.detach(), alpha=1.0 - self.decay)
            else:
                value.copy_(source)

    def state_dict(self) -> dict:
        return self.module.state_dict()


@dataclass
class ScalarEMA:
    decay: float = 0.92
    value: float | None = None

    def update(self, value: float) -> float:
        value = float(value)
        if self.value is None:
            self.value = value
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * value
        return self.value


def auxiliary_weight(base_weight: float, epoch: int, warmup_epochs: int) -> float:
    """Linearly ramp newly added heads after the stable objectives warm up."""

    if warmup_epochs <= 0:
        return float(base_weight)
    progress = min(1.0, max(0.0, (int(epoch) + 1) / float(warmup_epochs)))
    return float(base_weight) * progress

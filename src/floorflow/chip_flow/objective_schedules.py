"""Reusable objective schedules and scalar reductions for chip training."""

from __future__ import annotations

import argparse
import math

import torch


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def hierarchy_bbox_schedule(
    base_weight: float,
    decay_steps: int,
    step: int,
) -> float:
    """Linearly remove bbox alignment after its early-layout role is learned."""

    if decay_steps <= 0:
        return float(base_weight)
    progress = min(1.0, max(0.0, step / decay_steps))
    return float(base_weight) * (1.0 - progress)


def annealed_weight(
    start: float,
    final: float | None,
    step: int,
    anneal_steps: int,
    schedule: str = "linear",
) -> float:
    """Interpolate an objective weight from ``start`` to ``final``."""

    if schedule not in {"linear", "cosine"}:
        raise ValueError(f"unknown weight schedule: {schedule}")
    if step < 0 or anneal_steps < 0:
        raise ValueError("schedule step and anneal steps must be non-negative")
    if final is None or anneal_steps <= 0:
        return float(start)
    progress = min(1.0, max(0.0, float(step) / float(anneal_steps)))
    if schedule == "linear":
        factor = 1.0 - progress
    elif schedule == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(final) + (float(start) - float(final)) * factor


def delayed_ramp_weight(
    target: float,
    delay_steps: int,
    ramp_steps: int,
    step: int,
) -> float:
    """Hold a term inactive, then ramp it to its target weight."""

    if step < delay_steps:
        return 0.0
    if ramp_steps <= 0:
        return float(target)
    progress = min(1.0, (step - delay_steps) / float(ramp_steps))
    return float(target) * progress


def scheduled_proxy_weight(args: argparse.Namespace, name: str, step: int) -> float:
    """Resolve an optional final proxy weight and apply its delayed ramp."""

    target = float(getattr(args, f"{name}_weight", 0.0))
    final = getattr(args, f"{name}_final_weight", None)
    if final is not None:
        target = float(final)
    return delayed_ramp_weight(
        target,
        int(getattr(args, f"{name}_delay_steps", 0)),
        int(getattr(args, f"{name}_anneal_steps", 0)),
        step,
    )


def stage_learning_rate(
    base_lr: float,
    local_step: int,
    stage_steps: int,
    warmup_steps: int,
) -> float:
    """Cosine-decay one isolated stage with a local warmup."""

    if base_lr <= 0.0 or stage_steps <= 0 or warmup_steps < 0:
        raise ValueError("invalid stage learning-rate schedule")
    local_step = max(0, int(local_step))
    warmup_scale = (
        1.0
        if warmup_steps == 0
        else min(1.0, (local_step + 1) / float(warmup_steps))
    )
    progress = min(1.0, local_step / float(stage_steps))
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * warmup_scale * decay


def top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Return the mean of the highest-error fraction of case values."""

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("tail fraction must be in (0, 1]")
    flat = values.reshape(-1)
    count = max(1, math.ceil(flat.numel() * float(fraction)))
    return flat.topk(count, largest=True).values.mean()


__all__ = [
    "annealed_weight",
    "delayed_ramp_weight",
    "hierarchy_bbox_schedule",
    "learning_rate",
    "scheduled_proxy_weight",
    "stage_learning_rate",
    "top_fraction_mean",
]

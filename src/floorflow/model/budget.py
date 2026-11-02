"""Parameter, compute, and adaptive output-head budget utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn


class BudgetExceededError(ValueError):
    """Raised when a requested architecture cannot fit its resource envelope."""


def _odd_at_least(value: float) -> int:
    parsed = max(1, int(round(float(value))))
    return parsed if parsed % 2 == 1 else parsed + 1


def derive_kernel_candidates(
    resolution: tuple[int, int],
    target_feature_px: int,
    *,
    max_scales: int = 4,
) -> tuple[int, ...]:
    """Derive odd kernel candidates from input scale, not from a fixed tuple."""

    if len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError("resolution must contain two positive values")
    if target_feature_px <= 0:
        raise ValueError("target_feature_px must be positive")
    if max_scales <= 0:
        raise ValueError("max_scales must be positive")
    scale = min(resolution) / 256.0
    base = _odd_at_least(target_feature_px * scale)
    limit = max(base, _odd_at_least(min(resolution) / 2.0))
    candidates: list[int] = []
    for index in range(max_scales):
        value = min(limit, base + 2 * index)
        if value not in candidates:
            candidates.append(value)
    return tuple(candidates)


def estimate_output_head_params(
    in_channels: int,
    out_channels: int,
    kernel_sizes: Iterable[int],
) -> int:
    kernels = tuple(int(size) for size in kernel_sizes)
    router_channels = max(16, int(in_channels) // 2)
    branch_params = sum(
        int(in_channels) * size * size
        + int(in_channels)
        + int(in_channels) * int(out_channels)
        + int(out_channels)
        for size in kernels
    )
    router_params = (
        int(in_channels) * router_channels * 9
        + router_channels
        + router_channels * int(out_channels) * len(kernels)
        + int(out_channels) * len(kernels)
    )
    return branch_params + router_params


def estimate_output_head_macs(
    in_channels: int,
    out_channels: int,
    kernel_sizes: Iterable[int],
    resolution: tuple[int, int],
) -> int:
    kernels = tuple(int(size) for size in kernel_sizes)
    pixels = int(resolution[0]) * int(resolution[1])
    router_channels = max(16, int(in_channels) // 2)
    branch_macs = sum(
        pixels
        * (
            int(in_channels) * size * size
            + int(in_channels) * int(out_channels)
        )
        for size in kernels
    )
    router_macs = pixels * (
        int(in_channels) * router_channels * 9
        + router_channels * int(out_channels) * len(kernels)
    )
    return branch_macs + router_macs


def select_output_kernels(
    resolution: tuple[int, int],
    target_feature_px: int,
    in_channels: int,
    out_channels: int,
    *,
    max_params: int | None = None,
    min_scales: int = 2,
    max_scales: int = 4,
) -> tuple[int, ...]:
    """Choose the smallest useful multi-scale head that fits the budget."""

    if min_scales <= 0 or min_scales > max_scales:
        raise ValueError("require 0 < min_scales <= max_scales")
    candidates = derive_kernel_candidates(
        resolution,
        target_feature_px,
        max_scales=max_scales,
    )
    if max_params is None:
        return candidates[:max_scales]
    selected: list[int] = []
    for size in candidates:
        candidate = selected + [size]
        cost = estimate_output_head_params(in_channels, out_channels, candidate)
        if cost <= max_params:
            selected = candidate
        elif len(selected) >= min_scales:
            break
    if len(selected) < min_scales:
        minimum = estimate_output_head_params(
            in_channels,
            out_channels,
            candidates[:min_scales],
        )
        raise BudgetExceededError(
            f"output head needs {minimum} parameters for {min_scales} scales, "
            f"budget is {max_params}"
        )
    return tuple(selected)


def resolve_output_kernels(
    output_kernels: Iterable[int] | None,
    resolution: tuple[int, int],
    base_channels: int,
    out_channels: int,
) -> tuple[tuple[int, int], tuple[int, ...]]:
    resolution = tuple(int(value) for value in resolution)
    if len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError("resolution must contain two positive values")
    kernels = (
        select_output_kernels(resolution, 2, base_channels, out_channels)
        if output_kernels is None
        else tuple(int(value) for value in output_kernels)
    )
    if not kernels or any(value <= 0 or value % 2 == 0 for value in kernels):
        raise ValueError("output_kernels must contain positive odd sizes")
    return resolution, kernels


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def parameter_table(module: nn.Module) -> dict[str, int]:
    return {
        name: count_parameters(child)
        for name, child in module.named_children()
        if count_parameters(child) > 0
    }


@dataclass(frozen=True)
class ComputeProfile:
    macs: int
    modules: dict[str, int]


def profile_forward_macs(
    module: nn.Module,
    forward: Callable[[], object],
) -> ComputeProfile:
    """Measure Conv2d/Linear MACs from real tensor shapes via forward hooks."""

    totals: dict[str, int] = {}
    handles = []

    def conv_hook(name: str):
        def hook(layer, inputs, output) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            kernel_h, kernel_w = layer.kernel_size
            macs = (
                output.numel()
                * (layer.in_channels // layer.groups)
                * kernel_h
                * kernel_w
            )
            totals[name] = totals.get(name, 0) + int(macs)

        return hook

    def linear_hook(name: str):
        def hook(layer, inputs, output) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            macs = output.numel() * layer.in_features
            totals[name] = totals.get(name, 0) + int(macs)

        return hook

    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d):
            handles.append(child.register_forward_hook(conv_hook(name)))
        elif isinstance(child, nn.Linear):
            handles.append(child.register_forward_hook(linear_hook(name)))
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            forward()
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)
    return ComputeProfile(macs=sum(totals.values()), modules=totals)


__all__ = [
    "BudgetExceededError",
    "ComputeProfile",
    "count_parameters",
    "derive_kernel_candidates",
    "estimate_output_head_macs",
    "estimate_output_head_params",
    "parameter_table",
    "profile_forward_macs",
    "resolve_output_kernels",
    "select_output_kernels",
]

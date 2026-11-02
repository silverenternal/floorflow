"""Fail-closed gradient validation before optimizer updates."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def nonfinite_gradient_names(
    module: nn.Module,
) -> tuple[str, ...]:
    """Return parameter names whose gradients contain NaN or infinity."""

    names = []
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            names.append(name)
    return tuple(names)


def clip_gradients_fail_closed(
    module: nn.Module,
    max_norm: float,
    *,
    parameters: Iterable[torch.nn.Parameter] | None = None,
) -> float:
    """Reject non-finite gradients, then clip so infinity never becomes NaN."""

    invalid = nonfinite_gradient_names(module)
    if invalid:
        preview = ", ".join(invalid[:8])
        raise FloatingPointError(
            f"non-finite gradients in {len(invalid)} parameters: {preview}"
        )
    selected = module.parameters() if parameters is None else parameters
    norm = torch.nn.utils.clip_grad_norm_(selected, float(max_norm))
    if not torch.isfinite(norm):
        raise FloatingPointError("gradient norm is non-finite after clipping")
    return float(norm.detach())


__all__ = ["clip_gradients_fail_closed", "nonfinite_gradient_names"]

"""Parameter-level health checks for semantic chip model atoms."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def prefix_health(
    model: nn.Module,
    prefixes: tuple[str, ...],
    *,
    threshold: float = 1.0e-9,
) -> dict[str, float | int | bool]:
    """Summarize parameters and gradients owned by one atom's prefixes."""

    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    ]
    if not selected:
        return {
            "tensor_count": 0,
            "parameter_count": 0,
            "gradient_tensor_count": 0,
            "gradient_norm": 0.0,
            "gate_abs_max": 0.0,
            "finite": True,
            "active": False,
        }
    gate_parameters = [
        parameter
        for name, parameter in selected
        if name.endswith(("gate", "coordinate_gate.bias"))
    ]
    gradient_count = sum(parameter.grad is not None for _, parameter in selected)
    gradient_norm = max(
        (
            float(parameter.grad.detach().norm().cpu())
            if parameter.grad is not None
            else 0.0
        )
        for _, parameter in selected
    )
    gate_abs_max = max(
        (
            float(parameter.detach().abs().max().cpu())
            for parameter in gate_parameters
        ),
        default=0.0,
    )
    finite = all(
        bool(torch.isfinite(parameter.detach()).all())
        for _, parameter in selected
    )
    return {
        "tensor_count": len(selected),
        "parameter_count": sum(parameter.numel() for _, parameter in selected),
        "gradient_tensor_count": gradient_count,
        "gradient_norm": gradient_norm,
        "gate_abs_max": gate_abs_max,
        "finite": finite,
        "active": bool(
            finite
            and gradient_count > 0
            and gradient_norm > threshold
            and (not gate_parameters or gate_abs_max > threshold)
        ),
    }


def atom_health(
    model: nn.Module,
    specs: Mapping[str, tuple[str, ...]],
    *,
    threshold: float = 1.0e-9,
) -> dict[str, dict[str, float | int | bool]]:
    return {
        atom_id: prefix_health(model, prefixes, threshold=threshold)
        for atom_id, prefixes in specs.items()
    }


__all__ = ["atom_health", "prefix_health"]

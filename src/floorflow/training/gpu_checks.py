"""Fail-closed device checks that avoid unnecessary CUDA host synchronizations."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from floorflow.representation.tensor_checks import assert_true_async


def assert_finite_async(
    value: torch.Tensor,
    message: str,
) -> None:
    """Raise if a tensor is non-finite, using an asynchronous CUDA assertion."""

    condition = torch.isfinite(value).all()
    try:
        assert_true_async(condition, message)
    except ValueError as error:
        raise FloatingPointError(message) from error


def assert_parameters_finite(
    parameters: Iterable[torch.nn.Parameter],
    message: str,
) -> None:
    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return
    conditions = torch.stack(
        [torch.isfinite(gradient).all() for gradient in gradients]
    ).all()
    try:
        assert_true_async(conditions, message)
    except ValueError as error:
        raise FloatingPointError(message) from error


__all__ = ["assert_finite_async", "assert_parameters_finite"]

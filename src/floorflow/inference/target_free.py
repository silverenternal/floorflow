"""Auditable boundary between target-free generation and metric evaluation."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch

FORBIDDEN_INPUT_NAMES = frozenset(
    {
        "target",
        "targets",
        "ground_truth",
        "groundtruth",
        "gt",
        "gt_labels",
        "gt_mask",
        "oracle",
        "oracle_control",
        "oracle_controls",
        "reference",
    }
)


class TargetLeakError(ValueError):
    """Raised when a generation API exposes target-derived information."""


def _normalized(name: str) -> str:
    return name.strip().lower().replace("-", "_")


@dataclass(frozen=True)
class GenerationRequest:
    """The complete information available to the production generator."""

    footprint: torch.Tensor
    floor_ids: torch.Tensor | None = None
    style_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.footprint.ndim not in (3, 4):
            raise ValueError("footprint must have shape [B,H,W] or [B,1,H,W]")
        batch = self.footprint.shape[0]
        for name, value in (
            ("floor_ids", self.floor_ids),
            ("style_ids", self.style_ids),
        ):
            if value is not None and value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional batch tensor")
            if value is not None and value.shape[0] != batch:
                raise ValueError(f"{name} batch size does not match footprint")

    def as_kwargs(self) -> dict[str, torch.Tensor | None]:
        return {
            "footprint": self.footprint,
            "floor_ids": self.floor_ids,
            "style_ids": self.style_ids,
        }


def forbidden_parameters(function: Callable[..., Any]) -> tuple[str, ...]:
    """Return forbidden generation parameters exposed by a callable."""

    found = []
    for name in inspect.signature(function).parameters:
        normalized = _normalized(name)
        if normalized in FORBIDDEN_INPUT_NAMES:
            found.append(name)
            continue
        if any(
            token in normalized
            for token in (
                "target",
                "ground_truth",
                "oracle_control",
            )
        ):
            found.append(name)
    return tuple(found)


def validate_target_free_signature(function: Callable[..., Any]) -> None:
    """Reject a generation callable whose public API accepts target data."""

    forbidden = forbidden_parameters(function)
    if forbidden:
        joined = ", ".join(forbidden)
        raise TargetLeakError(
            f"generation callable {function.__qualname__} exposes target-derived "
            f"parameters: {joined}"
        )


def validate_generation_kwargs(kwargs: Mapping[str, Any]) -> None:
    leaked = sorted(
        name
        for name in kwargs
        if _normalized(name) in FORBIDDEN_INPUT_NAMES
        or "target" in _normalized(name)
    )
    if leaked:
        raise TargetLeakError(
            "target-derived generation arguments are forbidden: "
            + ", ".join(leaked)
        )


def target_free_generation(function: Callable[..., Any]) -> Callable[..., Any]:
    """Mark and enforce a callable as part of the target-free generation path."""

    validate_target_free_signature(function)

    @wraps(function)
    def audited(*args: Any, **kwargs: Any) -> Any:
        validate_generation_kwargs(kwargs)
        return function(*args, **kwargs)

    audited.target_free = True  # type: ignore[attr-defined]
    return audited


__all__ = [
    "FORBIDDEN_INPUT_NAMES",
    "GenerationRequest",
    "TargetLeakError",
    "forbidden_parameters",
    "target_free_generation",
    "validate_generation_kwargs",
    "validate_target_free_signature",
]

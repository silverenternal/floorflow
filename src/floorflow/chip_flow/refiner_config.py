"""Validation and parsing for the reusable Trinity closed-form refiner."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

from floorflow.chip_flow.hard_constraint_projector import project_hard_constraints

TERM_KEYS = ("overlap", "group", "mib", "boundary", "wl", "area")
NUMERIC_WEIGHT_KEYS = frozenset((*TERM_KEYS, "outline", "overlap_margin"))


def parse_refiner_weight_overrides(
    overrides: Iterable[str],
    *,
    base: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return a validated copy of the refiner weights with ``KEY=VALUE`` changes."""

    if base is None:
        from trinity.sampling.refine_closed import PAPER_WEIGHTS

        base = PAPER_WEIGHTS
    weights = dict(base)
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(
                f"refiner weight override must be KEY=VALUE, got {override!r}"
            )
        if key not in NUMERIC_WEIGHT_KEYS:
            allowed = ", ".join(sorted(NUMERIC_WEIGHT_KEYS))
            raise ValueError(f"unknown refiner weight {key!r}; expected one of {allowed}")
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"invalid refiner weight for {key!r}: {raw_value!r}") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"refiner weight {key!r} must be finite and non-negative")
        weights[key] = value
    return weights


def append_refiner_protocol(
    command: list[str],
    policy: dict[str, Any],
) -> None:
    """Append one validated refiner/legalizer protocol to an adapter command."""

    command.extend(
        [
            "--refiner-lr",
            str(policy.get("refiner_learning_rate", 1.0e-3)),
            "--refiner-schedule",
            str(policy.get("refiner_schedule", "linear")),
            "--refiner-optimizer",
            str(policy.get("refiner_optimizer", "adam")),
            "--refiner-chunk",
            str(policy.get("refiner_chunk", 5)),
            "--legalizer-time-limit-ms",
            str(policy.get("legalizer_time_limit_ms", 2000)),
        ]
    )
    for key, value in policy.get("refiner_weights", {}).items():
        command.extend(["--refiner-weight", f"{key}={value}"])
    command.append(
        "--refiner-shape"
        if policy.get("refiner_shape_update", False)
        else "--no-refiner-shape"
    )
    if policy.get("deterministic", False):
        command.append("--deterministic")


def project_after_refiner(
    latent: torch.Tensor,
    target_latent: torch.Tensor,
    constraints: torch.Tensor,
    areas: torch.Tensor,
    *,
    iterations: int,
    strength: float,
    boundary_axes_locked: bool = False,
    lock_boundary_corners: bool = True,
    sparse_pairs: bool = True,
    projection_strategy: str = "legacy",
    grid_size: int = 16,
) -> torch.Tensor:
    """Blend a refined latent toward the existing hard-projector output."""

    if iterations < 0:
        raise ValueError("post-refiner projector iterations must be non-negative")
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("post-refiner projector strength must be in [0, 1]")
    if iterations == 0 or strength == 0.0:
        return latent
    projected = project_hard_constraints(
        latent,
        target_latent,
        constraints,
        areas,
        iterations=iterations,
        boundary_axes_locked=boundary_axes_locked,
        lock_boundary_corners=lock_boundary_corners,
        sparse_pairs=sparse_pairs,
        projection_strategy=projection_strategy,
        grid_size=grid_size,
        collect_diagnostics=False,
    ).latent
    return torch.lerp(latent, projected, strength)


def project_after_refiner_batch(
    latent: torch.Tensor,
    batch: dict,
    model,
    *,
    iterations: int,
    strength: float,
) -> torch.Tensor:
    """Apply post-refiner projection with the model's deployment projector."""

    return project_after_refiner(
        latent,
        batch["target_latent"],
        batch["constraints"],
        batch["areas"],
        iterations=iterations,
        strength=strength,
        boundary_axes_locked=model.projector_boundary_axes,
        lock_boundary_corners=model.projector_boundary_lock_corners,
        sparse_pairs=model.projector_sparse_pairs,
        projection_strategy=model.projector_strategy,
        grid_size=model.projector_grid_size,
    )


def configure_chip_projector(
    model: torch.nn.Module,
    *,
    enabled: bool | None,
    iterations: int | None,
    boundary_axes: bool | None,
    boundary_lock_corners: bool | None,
    sparse_pairs: bool | None = None,
    projection_strategy: str | None = None,
    grid_size: int | None = None,
) -> torch.nn.Module:
    """Apply evaluator-side projector overrides to supported chip models."""

    if hasattr(model, "projector") and enabled is not None:
        model.projector = bool(enabled)
    if hasattr(model, "projector_iterations") and iterations is not None:
        model.projector_iterations = int(iterations)
    if hasattr(model, "projector_boundary_axes") and boundary_axes is not None:
        model.projector_boundary_axes = bool(boundary_axes)
    if (
        hasattr(model, "projector_boundary_lock_corners")
        and boundary_lock_corners is not None
    ):
        model.projector_boundary_lock_corners = bool(boundary_lock_corners)
    if hasattr(model, "projector_sparse_pairs") and sparse_pairs is not None:
        model.projector_sparse_pairs = bool(sparse_pairs)
    if hasattr(model, "projector_strategy") and projection_strategy is not None:
        model.projector_strategy = str(projection_strategy)
    if hasattr(model, "projector_grid_size") and grid_size is not None:
        model.projector_grid_size = int(grid_size)
    return model


def refiner_comparison_status(result_count: int, improvement: float) -> str:
    """Classify multi-setting sweeps without rejecting a single diagnostic."""

    if result_count <= 1:
        return "single_setting_diagnostic"
    return "pass" if improvement >= 0.05 else "refiner_lever_rejected"


__all__ = [
    "NUMERIC_WEIGHT_KEYS",
    "append_refiner_protocol",
    "configure_chip_projector",
    "parse_refiner_weight_overrides",
    "project_after_refiner",
    "project_after_refiner_batch",
    "refiner_comparison_status",
]

"""Source-manifold mixing for chip flow training."""

from __future__ import annotations

import math

import torch

from floorflow.chip_flow.objective_schedules import annealed_weight


def source_shelf_probability_at_step(args, step: int) -> float:
    """Return the scheduled deployment-source probability."""

    if step < 0:
        raise ValueError("source schedule step must be non-negative")
    start = float(getattr(args, "hierarchy_source_shelf_probability", 0.5))
    final_value = getattr(args, "hierarchy_source_shelf_probability_final", None)
    bounds = [start, float(final_value)] if final_value is not None else [start]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in bounds):
        raise ValueError("source shelf probabilities must be finite and in [0, 1]")
    anneal_steps = int(
        getattr(args, "hierarchy_source_shelf_probability_steps", 0)
    )
    if anneal_steps < 0:
        raise ValueError("source schedule steps must be non-negative")
    probability = annealed_weight(
        start,
        final_value,
        step,
        anneal_steps,
        str(getattr(args, "weight_schedule", "linear")),
    )
    maximum = float(
        getattr(args, "hierarchy_source_max_shelf_probability", 1.0)
    )
    if not math.isfinite(maximum) or not 0.0 <= maximum <= 1.0:
        raise ValueError("maximum shelf probability must be finite and in [0, 1]")
    return min(probability, maximum)


def mix_random_and_shelf_source(
    shelf_source: torch.Tensor,
    *,
    generator: torch.Generator | None,
    shelf_probability: float,
) -> torch.Tensor:
    """Mix whole-case shelf and random sources with a fixed probability."""

    if not 0.0 <= shelf_probability <= 1.0:
        raise ValueError("shelf probability must be in [0, 1]")
    random_source = torch.randn(
        shelf_source.shape,
        device=shelf_source.device,
        dtype=shelf_source.dtype,
        generator=generator,
    )
    shelf_mask = torch.rand(
        shelf_source.shape[0],
        device=shelf_source.device,
        generator=generator,
    ).lt(float(shelf_probability))
    return torch.where(
        shelf_mask[:, None, None],
        shelf_source,
        random_source,
    )


__all__ = ["mix_random_and_shelf_source", "source_shelf_probability_at_step"]

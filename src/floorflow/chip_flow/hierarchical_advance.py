"""Coarse-commit-aware Euler advancement for hierarchy states."""

from __future__ import annotations

import torch

from floorflow.chip_flow.hierarchy_state import HierarchyPrediction, HierarchyState


def advance_hierarchy_state(
    state: HierarchyState,
    prediction: HierarchyPrediction,
    denominator: torch.Tensor,
    *,
    step_size: float,
    port_mask: torch.Tensor,
    coarse_frozen: bool = False,
    coarse_anchor_masks: tuple[torch.Tensor, torch.Tensor] | None = None,
    coarse_anchor_strength: float = 0.0,
    coarse_anchor_max_step: float = 0.1,
) -> HierarchyState:
    """Advance L2/L3 while optionally freezing committed L0/L1 state."""

    if not 0.0 <= coarse_anchor_strength <= 1.0:
        raise ValueError("coarse anchor strength must be in [0, 1]")
    if coarse_anchor_max_step < 0.0:
        raise ValueError("coarse anchor max step must be non-negative")

    def advance(
        value: torch.Tensor,
        target: torch.Tensor,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        view = denominator.view(-1, *([1] * (value.ndim - 1)))
        delta = (target - value) / view * step_size
        if anchor_mask is not None and coarse_anchor_strength > 0.0:
            bounded = delta.clamp(
                -coarse_anchor_max_step,
                coarse_anchor_max_step,
            )
            bounded = bounded * coarse_anchor_strength
            delta = torch.where(
                anchor_mask[..., None],
                bounded,
                delta,
            )
        return value + delta

    global_mask = (
        coarse_anchor_masks[0] if coarse_anchor_masks is not None else None
    )
    structural_mask = (
        coarse_anchor_masks[1] if coarse_anchor_masks is not None else None
    )

    return HierarchyState(
        global_state=(
            state.global_state
            if coarse_frozen
            else advance(
                state.global_state,
                prediction.global_state,
                global_mask,
            )
        ),
        structural_state=(
            state.structural_state
            if coarse_frozen
            else advance(
                state.structural_state,
                prediction.structural_state,
                structural_mask,
            )
        ),
        block_state=advance(state.block_state, prediction.block_state),
        port_state=(
            advance(state.port_state, prediction.port_state)
            * port_mask[..., None]
        ),
        structural_moment_state=(
            None
            if state.structural_moment_state is None
            or prediction.structural_moment_state is None
            else (
                state.structural_moment_state
                if coarse_frozen
                else advance(
                    state.structural_moment_state,
                    prediction.structural_moment_state,
                    structural_mask,
                )
            )
        ),
    )


def coarse_commit_threshold(
    fraction: float,
    sampling_steps: int,
) -> int:
    """Return the first step at which coarse states remain committed."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("coarse commit fraction must be in [0, 1]")
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive")
    if fraction == 0.0:
        return 0
    if sampling_steps == 1:
        return 1
    return min(sampling_steps - 1, max(1, round(fraction * sampling_steps)))


def coarse_stability_masks(
    state: HierarchyState,
    prediction: HierarchyPrediction,
    *,
    tolerance: float,
    structural_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return state-stable global and structural slots for bounded anchoring."""

    if tolerance < 0.0:
        raise ValueError("coarse stability tolerance must be non-negative")
    global_change = (
        prediction.global_state - state.global_state
    ).abs().mean(dim=-1)
    structural_change = (
        prediction.structural_state - state.structural_state
    ).abs().amax(dim=-1)
    return (
        global_change.le(tolerance),
        structural_change.le(tolerance) & structural_mask,
    )


__all__ = [
    "advance_hierarchy_state",
    "coarse_commit_threshold",
    "coarse_stability_masks",
]

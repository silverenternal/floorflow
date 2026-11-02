"""Differentiable per-step chip projector with a final scale_pack guarantee."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import prepare_conditioned_projection
from floorflow.chip_flow.projector_geometry import (
    boundary_violation_count,
    candidate_overlap_pairs,
    joint_conflict_energy,
    mib_shape_violation_count,
    overlap_pair_count,
)
from floorflow.chip_flow.projector_geometry import (
    project_boundary as _project_boundary,
)
from floorflow.chip_flow.projector_geometry import (
    separate_overlaps as _separate_overlaps,
)

PROJECTION_STRATEGIES = ("legacy", "conflict_aware")


@dataclass(frozen=True)
class ProjectionOutput:
    latent: torch.Tensor
    boxes: torch.Tensor
    remaining_overlap_pairs: int
    boundary_violations: int
    mib_shape_violations: int
    max_area_error: float
    fixed_shape_error: float
    preplaced_max_error: float
    candidate_pair_count: int = 0
    joint_conflict_energy: float = 0.0


def _overlap_pairs(boxes: torch.Tensor) -> int:
    return overlap_pair_count(boxes)


def _boundary_violation_count(
    boxes: torch.Tensor,
    constraints: torch.Tensor,
) -> int:
    return boundary_violation_count(boxes, constraints)


def _mib_shape_violations(
    boxes: torch.Tensor,
    constraints: torch.Tensor,
) -> int:
    return mib_shape_violation_count(boxes, constraints)


def _replace_centers(
    latent: torch.Tensor,
    boxes: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    centers = boxes[..., :2] + boxes[..., 2:] / 2.0
    return torch.stack(
        [
            centers[..., 0] / scale[:, None],
            centers[..., 1] / scale[:, None],
            latent[..., 2],
        ],
        dim=-1,
    )


def _projection_order(
    latent: torch.Tensor,
    plan,
    constraints: torch.Tensor,
    areas: torch.Tensor,
    *,
    overlap_first: bool,
    boundary_axes_locked: bool,
    lock_boundary_corners: bool,
    sparse_pairs: bool,
    grid_size: int,
) -> torch.Tensor:
    projected = latent
    if not overlap_first:
        projected = plan.apply(projected)
        projected = _project_boundary(projected, constraints, areas)
    boxes = decode_boxes_batch(projected, areas)
    movable = ~constraints[..., 1].ne(0)
    moved = _separate_overlaps(
        boxes,
        movable,
        constraints,
        boundary_axes_locked=boundary_axes_locked,
        lock_boundary_corners=lock_boundary_corners,
        sparse_pairs=sparse_pairs,
        grid_size=grid_size,
    )
    projected = _replace_centers(projected, moved, areas)
    projected = plan.apply(projected)
    return _project_boundary(projected, constraints, areas)


def _conflict_aware_step(
    latent: torch.Tensor,
    plan,
    constraints: torch.Tensor,
    areas: torch.Tensor,
    *,
    boundary_axes_locked: bool,
    lock_boundary_corners: bool,
    sparse_pairs: bool,
    grid_size: int,
) -> torch.Tensor:
    boundary_first = _projection_order(
        latent,
        plan,
        constraints,
        areas,
        overlap_first=False,
        boundary_axes_locked=boundary_axes_locked,
        lock_boundary_corners=lock_boundary_corners,
        sparse_pairs=sparse_pairs,
        grid_size=grid_size,
    )
    overlap_first = _projection_order(
        latent,
        plan,
        constraints,
        areas,
        overlap_first=True,
        boundary_axes_locked=boundary_axes_locked,
        lock_boundary_corners=lock_boundary_corners,
        sparse_pairs=sparse_pairs,
        grid_size=grid_size,
    )
    boundary_energy = joint_conflict_energy(
        boundary_first,
        constraints,
        areas,
        grid_size=grid_size,
    ).detach()
    overlap_energy = joint_conflict_energy(
        overlap_first,
        constraints,
        areas,
        grid_size=grid_size,
    ).detach()
    choose_boundary_first = boundary_energy <= overlap_energy
    return torch.where(
        choose_boundary_first[:, None, None],
        boundary_first,
        overlap_first,
    )


def project_hard_constraints(
    latent: torch.Tensor,
    target_latent: torch.Tensor,
    constraints: torch.Tensor,
    areas: torch.Tensor,
    *,
    iterations: int = 8,
    boundary_axes_locked: bool = False,
    lock_boundary_corners: bool = True,
    sparse_pairs: bool = True,
    projection_strategy: str = "legacy",
    grid_size: int = 16,
    collect_diagnostics: bool = True,
) -> ProjectionOutput:
    """Approximate hard geometry while preserving differentiability."""

    projected = project_hard_constraints_latent(
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
    )
    boxes = decode_boxes_batch(projected, areas)
    if not collect_diagnostics:
        return ProjectionOutput(
            latent=projected,
            boxes=boxes,
            remaining_overlap_pairs=0,
            boundary_violations=0,
            mib_shape_violations=0,
            max_area_error=0.0,
            fixed_shape_error=0.0,
            preplaced_max_error=0.0,
        )
    target_boxes = decode_boxes_batch(target_latent, areas)
    soft = ~constraints[..., :2].ne(0).any(dim=-1)
    fixed = constraints[..., 0].ne(0)
    preplaced = constraints[..., 1].ne(0)
    area_error = (boxes[..., 2] * boxes[..., 3] - areas).abs()[soft]
    fixed_error = (boxes[..., 2:] - target_boxes[..., 2:]).abs()[fixed]
    preplaced_error = (boxes - target_boxes).abs()[preplaced]
    candidates = candidate_overlap_pairs(boxes, grid_size=grid_size)[0].numel()
    return ProjectionOutput(
        latent=projected,
        boxes=boxes,
        remaining_overlap_pairs=_overlap_pairs(boxes),
        boundary_violations=_boundary_violation_count(boxes, constraints),
        mib_shape_violations=_mib_shape_violations(boxes, constraints),
        max_area_error=(
            float(area_error.detach().max()) if area_error.numel() else 0.0
        ),
        fixed_shape_error=(
            float(fixed_error.detach().max()) if fixed_error.numel() else 0.0
        ),
        preplaced_max_error=(
            float(preplaced_error.detach().max())
            if preplaced_error.numel()
            else 0.0
        ),
        candidate_pair_count=int(candidates),
        joint_conflict_energy=float(
            joint_conflict_energy(
                projected,
                constraints,
                areas,
                grid_size=grid_size,
            )
            .mean()
            .detach()
        ),
    )


def project_hard_constraints_latent(
    latent: torch.Tensor,
    target_latent: torch.Tensor,
    constraints: torch.Tensor,
    areas: torch.Tensor,
    *,
    iterations: int = 8,
    boundary_axes_locked: bool = False,
    lock_boundary_corners: bool = True,
    sparse_pairs: bool = True,
    projection_strategy: str = "legacy",
    grid_size: int = 16,
) -> torch.Tensor:
    """Run an out-of-place projector loop suitable for endpoint training."""

    if projection_strategy not in PROJECTION_STRATEGIES:
        raise ValueError(f"unknown projection strategy: {projection_strategy}")
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    projected = latent
    plan = prepare_conditioned_projection(target_latent, constraints)
    for _ in range(iterations):
        if projection_strategy == "conflict_aware":
            projected = _conflict_aware_step(
                projected,
                plan,
                constraints,
                areas,
                boundary_axes_locked=boundary_axes_locked,
                lock_boundary_corners=lock_boundary_corners,
                sparse_pairs=sparse_pairs,
                grid_size=grid_size,
            )
        else:
            projected = _projection_order(
                projected,
                plan,
                constraints,
                areas,
                overlap_first=False,
                boundary_axes_locked=boundary_axes_locked,
                lock_boundary_corners=lock_boundary_corners,
                sparse_pairs=sparse_pairs,
                grid_size=grid_size,
            )
    if projection_strategy == "legacy":
        projected = plan.apply(projected)
        projected = _project_boundary(projected, constraints, areas)
    return projected


@lru_cache(maxsize=4)
def _compiled_projector(mode: str):
    return torch.compile(project_hard_constraints_latent, mode=mode)


def compiled_project_hard_constraints_latent(
    latent: torch.Tensor,
    target_latent: torch.Tensor,
    constraints: torch.Tensor,
    areas: torch.Tensor,
    *,
    iterations: int = 8,
    boundary_axes_locked: bool = False,
    lock_boundary_corners: bool = True,
    sparse_pairs: bool = True,
    projection_strategy: str = "legacy",
    grid_size: int = 16,
    mode: str = "reduce-overhead",
) -> torch.Tensor:
    """Opt-in compiled projector path with diagnostics intentionally excluded."""

    projector = _compiled_projector(mode)
    return projector(
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
    )


def corrected_velocity(
    velocity: torch.Tensor,
    latent: torch.Tensor,
    target_latent: torch.Tensor,
    constraints: torch.Tensor,
    areas: torch.Tensor,
    *,
    step_size: float,
    strength: float = 1.0,
    iterations: int = 8,
    sparse_pairs: bool = True,
    projection_strategy: str = "legacy",
    grid_size: int = 16,
) -> tuple[torch.Tensor, ProjectionOutput]:
    """Pull an Euler step toward the projected feasible state."""

    candidate = latent + velocity * step_size
    projected = project_hard_constraints(
        candidate,
        target_latent,
        constraints,
        areas,
        iterations=iterations,
        sparse_pairs=sparse_pairs,
        projection_strategy=projection_strategy,
        grid_size=grid_size,
    )
    correction = (projected.latent - candidate) / step_size
    return velocity + strength * correction, projected


__all__ = [
    "PROJECTION_STRATEGIES",
    "ProjectionOutput",
    "compiled_project_hard_constraints_latent",
    "corrected_velocity",
    "project_hard_constraints",
    "project_hard_constraints_latent",
]

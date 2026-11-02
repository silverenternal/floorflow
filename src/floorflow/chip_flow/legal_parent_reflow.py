"""Bounded partial re-noising around a legal chip parent."""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from floorflow.chip_flow.hard_constraint_projector import project_hard_constraints
from floorflow.chip_flow.hierarchical_flow_runtime import (
    hierarchy_source_state,
    integrate_hierarchy_step,
)
from floorflow.chip_flow.model import decode_boxes_batch


@torch.no_grad()
def critical_block_mask(
    boxes: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
    constraints: torch.Tensor,
    n_b2b: int,
    n_p2b: int,
    *,
    top_fraction: float = 0.2,
) -> torch.Tensor:
    """Select movable blocks with the largest active net and pin pressure."""

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    count = boxes.shape[0]
    centers = boxes[:, :2] + boxes[:, 2:] * 0.5
    pressure = boxes.new_zeros(count)
    if n_b2b > 0 and b2b.numel():
        edges = b2b[:n_b2b]
        left = edges[:, 0].long().clamp(0, count - 1)
        right = edges[:, 1].long().clamp(0, count - 1)
        span = (centers[left] - centers[right]).abs().sum(dim=-1)
        weight = span * edges[:, 2].clamp_min(0.0)
        pressure.scatter_add_(0, left, weight)
        pressure.scatter_add_(0, right, weight)
    if n_p2b > 0 and p2b.numel() and pins.numel():
        edges = p2b[:n_p2b]
        pin_id = edges[:, 0].long().clamp(0, pins.shape[0] - 1)
        block_id = edges[:, 1].long().clamp(0, count - 1)
        distance = (pins[pin_id].to(boxes.device) - centers[block_id]).abs().sum(-1)
        pressure.scatter_add_(0, block_id, distance * edges[:, 2].clamp_min(0.0))
    movable = ~constraints[:count, :2].ne(0).any(dim=-1)
    available = int(movable.sum())
    if available == 0:
        return movable
    selected = max(1, min(available, math.ceil(float(top_fraction) * count)))
    scores = pressure.masked_fill(~movable, float("-inf"))
    indices = torch.topk(scores, k=selected).indices
    mask = torch.zeros(count, dtype=torch.bool, device=boxes.device)
    mask[indices] = True
    return mask


@torch.no_grad()
def partial_renoise_parent(
    model,
    batch: dict,
    parent_latent: torch.Tensor,
    update_mask: torch.Tensor,
    generator: torch.Generator,
    *,
    noise_fraction: float = 0.25,
    steps: int = 2,
    projector_iterations: int = 8,
    projector_boundary_axes: bool = False,
    projector_boundary_lock_corners: bool = True,
    projector_sparse_pairs: bool = True,
    projector_strategy: str = "legacy",
    projector_grid_size: int = 16,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Re-noise critical blocks and integrate the existing flow back to t=1."""

    if parent_latent.ndim != 3 or parent_latent.shape[-1] != 3:
        raise ValueError("parent_latent must have shape (B, N, 3)")
    if update_mask.shape != parent_latent.shape[:2]:
        raise ValueError("update_mask must match parent batch and block dimensions")
    if not 0.0 < noise_fraction < 1.0:
        raise ValueError("noise_fraction must be in (0, 1)")
    if steps <= 0:
        raise ValueError("steps must be positive")
    context = model.prepare_context(batch)
    parent_state = hierarchy_source_state(
        parent_latent.float(),
        context,
        generator,
        source_mode="derived",
        batch=batch,
    )
    noise = torch.randn(
        parent_latent.shape,
        device=parent_latent.device,
        dtype=parent_latent.dtype,
        generator=generator,
    )
    start_time = float(noise_fraction)
    noisy_block = (1.0 - start_time) * noise + start_time * parent_latent.float()
    block_state = torch.where(
        update_mask[..., None],
        noisy_block,
        parent_latent.float(),
    )
    state = replace(parent_state, block_state=block_state)
    step_size = (1.0 - start_time) / steps
    for step in range(steps):
        time = torch.full(
            (parent_latent.shape[0],),
            start_time + step * step_size,
            device=parent_latent.device,
            dtype=parent_latent.dtype,
        )
        state, _ = integrate_hierarchy_step(
            model,
            state,
            time,
            context,
            batch,
            step_size=step_size,
            projector=True,
            projector_iterations=projector_iterations,
            projector_boundary_axes=projector_boundary_axes,
            projector_boundary_lock_corners=projector_boundary_lock_corners,
            projector_sparse_pairs=projector_sparse_pairs,
            projector_strategy=projector_strategy,
            projector_grid_size=projector_grid_size,
            collect_projector_diagnostics=False,
        )
        state = replace(
            state,
            block_state=torch.where(
                update_mask[..., None],
                state.block_state,
                parent_latent.float(),
            ),
        )
    projection = project_hard_constraints(
        state.block_state,
        parent_latent.float(),
        batch["constraints"],
        batch["areas"],
        iterations=projector_iterations,
        boundary_axes_locked=projector_boundary_axes,
        lock_boundary_corners=projector_boundary_lock_corners,
        sparse_pairs=projector_sparse_pairs,
        projection_strategy=projector_strategy,
        grid_size=projector_grid_size,
    )
    final_latent = torch.where(
        update_mask[..., None],
        projection.latent,
        parent_latent.float(),
    )
    audit = project_hard_constraints(
        final_latent,
        parent_latent.float(),
        batch["constraints"],
        batch["areas"],
        iterations=0,
        boundary_axes_locked=projector_boundary_axes,
        lock_boundary_corners=projector_boundary_lock_corners,
        sparse_pairs=projector_sparse_pairs,
        projection_strategy=projector_strategy,
        grid_size=projector_grid_size,
    )
    parent_boxes = decode_boxes_batch(parent_latent.float(), batch["areas"])
    reflow_boxes = decode_boxes_batch(final_latent, batch["areas"])
    valid = batch.get("n")
    if valid is not None:
        valid_tensor = torch.as_tensor(valid, device=parent_boxes.device)
        if valid_tensor.ndim == 0:
            valid_mask = torch.ones(
                update_mask.shape,
                dtype=torch.bool,
                device=parent_boxes.device,
            )
        else:
            rows = torch.arange(
                parent_boxes.shape[1],
                device=parent_boxes.device,
            )[None]
            valid_mask = rows < valid_tensor.long()[:, None]
    else:
        valid_mask = torch.ones_like(update_mask)
    displacement = torch.linalg.vector_norm(
        reflow_boxes[..., :2] - parent_boxes[..., :2],
        dim=-1,
    )
    outside = displacement.masked_fill(update_mask | ~valid_mask, 0.0)
    inside = displacement.masked_fill(~update_mask | ~valid_mask, 0.0)
    counts = update_mask.sum(dim=1).clamp_min(1)
    diagnostics = {
        "critical_blocks": update_mask.sum(dim=1).float(),
        "critical_displacement_mean": inside.sum(dim=1) / counts,
        "noncritical_displacement_mean": outside.sum(dim=1) / (
            (~update_mask & valid_mask).sum(dim=1).clamp_min(1)
        ),
        "remaining_overlap_pairs": torch.tensor(
            audit.remaining_overlap_pairs,
            device=parent_latent.device,
        ),
        "boundary_violations": torch.tensor(
            audit.boundary_violations,
            device=parent_latent.device,
        ),
        "mib_shape_violations": torch.tensor(
            audit.mib_shape_violations,
            device=parent_latent.device,
        ),
    }
    return final_latent, diagnostics


__all__ = ["critical_block_mask", "partial_renoise_parent"]

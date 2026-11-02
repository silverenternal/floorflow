"""Six differentiable chip-floorplanning terms matching Trinity's physics.md.

Boxes are ``[N, 4]`` tensors ``(x, y, w, h)`` in absolute units.  Instances
carry the FloorSet constraint columns ``[N, 5]`` = fixed, preplaced, MIB group,
cluster id, boundary code.  Every term is zero iff its constraint holds.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree


def _scale(areas: torch.Tensor) -> torch.Tensor:
    return (areas.sum() * 0.0 + areas.float()).sum().sqrt().clamp_min(1.0e-6)


def overlap_penalty(boxes: torch.Tensor, areas: torch.Tensor | None = None) -> torch.Tensor:
    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = x0 + boxes[:, 2], y0 + boxes[:, 3]
    left = torch.maximum(x0[:, None], x0[None, :])
    right = torch.minimum(x1[:, None], x1[None, :])
    top = torch.maximum(y0[:, None], y0[None, :])
    bottom = torch.minimum(y1[:, None], y1[None, :])
    intersection = (right - left).clamp_min(0.0) * (bottom - top).clamp_min(0.0)
    mask = ~torch.eye(len(boxes), dtype=torch.bool, device=boxes.device)
    total = intersection[mask].sum() * 0.5
    area_scale = (
        areas if areas is not None else boxes[:, 2] * boxes[:, 3]
    )
    return total / _scale(area_scale).pow(2)


def _gap(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    squeeze = left.ndim == 1
    if squeeze:
        left = left[None]
        right = right[None]
    gap_x = torch.maximum(
        left[:, 0], right[:, 0]
    ) - torch.minimum(left[:, 0] + left[:, 2], right[:, 0] + right[:, 2])
    gap_y = torch.maximum(
        left[:, 1], right[:, 1]
    ) - torch.minimum(left[:, 1] + left[:, 3], right[:, 1] + right[:, 3])
    edge_connected = ((gap_x <= 0) & (gap_y < 0)) | (
        (gap_y <= 0) & (gap_x < 0)
    )
    result = torch.where(
        edge_connected,
        torch.zeros_like(gap_x),
        torch.maximum(gap_x, gap_y).clamp_min(1.0e-3),
    )
    return result[0] if squeeze else result


def group_penalty(
    boxes: torch.Tensor,
    cluster_ids: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    return group_penalty_batch(
        boxes[None],
        cluster_ids[None],
        areas[None],
    )[0]


def group_penalty_batch(
    boxes: torch.Tensor,
    cluster_ids: torch.Tensor,
    areas: torch.Tensor,
    *,
    workers: int = 1,
) -> torch.Tensor:
    """Return exact per-case grouping penalties with one batched transfer.

    Gap matrices are prepared on the input device. SciPy's exact sparse MST is
    retained as the CPU boundary because its dense-zero semantics define the
    historical protocol; replacing it would silently change reported metrics.
    """

    if (
        boxes.ndim != 3
        or cluster_ids.shape != boxes.shape[:2]
        or areas.shape != boxes.shape[:2]
    ):
        raise ValueError("group penalty expects boxes [B,N,4], ids [B,N], areas [B,N]")
    if boxes.shape[1] < 2:
        return boxes.sum(dim=(1, 2)) * 0.0
    ids_cpu = cluster_ids.detach().to(dtype=torch.long, device="cpu").numpy()
    gaps_cpu = _gap_matrix_batch(boxes).detach().cpu().numpy()
    jobs = []
    for group in np.unique(ids_cpu[ids_cpu > 0]):
        member = ids_cpu == group
        for row in np.flatnonzero(member.any(axis=1)):
            local = np.flatnonzero(member[row])
            if local.size >= 2:
                jobs.append((int(row), local))

    def solve(job: tuple[int, np.ndarray]) -> tuple[int, float]:
        row, local = job
        gap = gaps_cpu[row][np.ix_(local, local)]
        return row, float(minimum_spanning_tree(gap).sum())

    worker_count = max(1, min(int(workers), len(jobs)))
    if worker_count == 1:
        solved = [solve(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            solved = list(executor.map(solve, jobs))
    totals_cpu = np.zeros(boxes.shape[0], dtype=np.float64)
    for row, value in solved:
        totals_cpu[row] += value
    totals = torch.from_numpy(totals_cpu).to(
        device=boxes.device,
        dtype=boxes.dtype,
    )
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    return boxes.sum(dim=(1, 2)) * 0.0 + totals / scale


def _gap_matrix_batch(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0 = boxes[..., 0], boxes[..., 1]
    x1, y1 = x0 + boxes[..., 2], y0 + boxes[..., 3]
    gap_x = torch.maximum(x0[:, :, None], x0[:, None, :]) - torch.minimum(
        x1[:, :, None],
        x1[:, None, :],
    )
    gap_y = torch.maximum(y0[:, :, None], y0[:, None, :]) - torch.minimum(
        y1[:, :, None],
        y1[:, None, :],
    )
    edge_connected = ((gap_x <= 0) & (gap_y < 0)) | (
        (gap_y <= 0) & (gap_x < 0)
    )
    return torch.where(
        edge_connected,
        torch.zeros_like(gap_x),
        torch.maximum(gap_x, gap_y).clamp_min(1.0e-3),
    )


def mib_penalty(
    boxes: torch.Tensor,
    mib_ids: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    log_width = boxes[:, 2].clamp_min(1.0e-6).log()
    log_height = boxes[:, 3].clamp_min(1.0e-6).log()
    total = log_width.sum() * 0.0
    for group in torch.unique(mib_ids):
        if float(group) <= 0.0:
            continue
        members = torch.nonzero(mib_ids == group, as_tuple=False).flatten()
        if members.numel() < 2:
            continue
        total = total + (
            (log_width[members] - log_width[members].mean()).abs()
            + (log_height[members] - log_height[members].mean()).abs()
        ).sum()
    del areas
    return total


def boundary_penalty(
    boxes: torch.Tensor,
    boundary_codes: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    x_min = boxes[:, 0].min()
    y_min = boxes[:, 1].min()
    x_max = (boxes[:, 0] + boxes[:, 2]).max()
    y_max = (boxes[:, 1] + boxes[:, 3]).max()
    codes = boundary_codes.to(dtype=torch.long, device=boxes.device)
    x, y, w, h = boxes.unbind(dim=1)
    left = (codes & 1).to(boxes.dtype) * (x - x_min).abs()
    right = (codes & 2).to(boxes.dtype) * (x_max - x - w).abs()
    top = (codes & 4).to(boxes.dtype) * (y_max - y - h).abs()
    bottom = (codes & 8).to(boxes.dtype) * (y - y_min).abs()
    return (left + right + top + bottom).sum() / _scale(areas)


def wirelength_objective(
    boxes: torch.Tensor,
    b2b_edges: torch.Tensor,
    p2b_edges: torch.Tensor,
    pins: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    centers = boxes[:, :2] + boxes[:, 2:] * 0.5
    total = boxes[:, 0].sum() * 0.0
    weight_sum = boxes[:, 0].sum() * 0.0
    if b2b_edges.numel():
        left = b2b_edges[:, 0].long()
        right = b2b_edges[:, 1].long()
        weights = (
            b2b_edges[:, 2]
            if b2b_edges.shape[1] > 2
            else torch.ones(len(b2b_edges), device=boxes.device)
        )
        total = total + (
            (centers[left] - centers[right]).abs().sum(dim=1) * weights
        ).sum()
        weight_sum = weight_sum + weights.sum()
    if p2b_edges.numel():
        pin_index = p2b_edges[:, 0].long()
        block_index = p2b_edges[:, 1].long()
        weights = (
            p2b_edges[:, 2]
            if p2b_edges.shape[1] > 2
            else torch.ones(len(p2b_edges), device=boxes.device)
        )
        target = pins[pin_index].to(boxes.device, dtype=boxes.dtype)
        total = total + (
            (centers[block_index] - target).abs().sum(dim=1) * weights
        ).sum()
        weight_sum = weight_sum + weights.sum()
    return total / (
        _scale(areas) * weight_sum.clamp_min(1.0)
    )


def area_objective(boxes: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
    width = boxes[:, 0].max() - boxes[:, 0].min()
    height = (boxes[:, 1] + boxes[:, 3]).max() - boxes[:, 1].min()
    return width * height / _scale(areas).pow(2)


def chip_physics_loss(
    boxes: torch.Tensor,
    areas: torch.Tensor,
    constraints: torch.Tensor,
    b2b_edges: torch.Tensor,
    p2b_edges: torch.Tensor,
    pins: torch.Tensor,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Return every term plus the weighted total (Trinity-compatible)."""

    weights = weights or {
        "overlap": 0.01,
        "group": 0.01,
        "mib": 0.01,
        "boundary": 0.01,
        "wl": 0.01,
        "area": 0.01,
    }
    terms = {
        "overlap": overlap_penalty(boxes, areas),
        "group": group_penalty(boxes, constraints[:, 3], areas),
        "mib": mib_penalty(boxes, constraints[:, 2], areas),
        "boundary": boundary_penalty(boxes, constraints[:, 4], areas),
        "wl": wirelength_objective(boxes, b2b_edges, p2b_edges, pins, areas),
        "area": area_objective(boxes, areas),
    }
    total = sum(
        float(weights.get(name, 0.0)) * value
        for name, value in terms.items()
    )
    return {**terms, "total": total}


__all__ = [
    "area_objective",
    "boundary_penalty",
    "chip_physics_loss",
    "group_penalty",
    "group_penalty_batch",
    "mib_penalty",
    "overlap_penalty",
    "wirelength_objective",
]

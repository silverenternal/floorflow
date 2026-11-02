"""Geometry helpers that use the known exterior contour as a hard condition."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.classes import BACKGROUND, WALL


def _as_bchw(footprint: torch.Tensor) -> torch.Tensor:
    if footprint.ndim == 3:
        return footprint[:, None]
    if footprint.ndim == 4:
        return footprint
    raise ValueError(f"footprint must be [B,H,W] or [B,1,H,W], got {tuple(footprint.shape)}")


def footprint_band(footprint: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Return the known exterior-wall band of the input contour."""

    footprint_bchw = _as_bchw(footprint).float()
    kernel_size = max(1, int(kernel_size))
    padding = kernel_size // 2
    eroded = -F.max_pool2d(-footprint_bchw, kernel_size, stride=1, padding=padding)
    return (footprint_bchw - eroded).clamp(0.0, 1.0)


def interior_mask(footprint: torch.Tensor, boundary_width: int = 7) -> torch.Tensor:
    """Interior pixels for which the model is responsible."""

    footprint_bchw = _as_bchw(footprint).float()
    return (footprint_bchw - footprint_band(footprint_bchw, boundary_width)).clamp(0.0, 1.0)


def compose_known_exterior(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    *,
    boundary_width: int = 5,
    opening_ids: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Composite the known envelope while preserving walls' door/window openings."""

    labels = labels.long().clone()
    footprint_bchw = _as_bchw(footprint)
    outside = footprint_bchw[:, 0] <= 0.5
    boundary = footprint_band(footprint_bchw, boundary_width)[:, 0] > 0.5
    labels[outside] = BACKGROUND
    preserve = torch.zeros_like(boundary)
    if opening_ids:
        ids = torch.tensor(tuple(int(value) for value in opening_ids), device=labels.device)
        preserve = torch.isin(labels, ids)
    labels[boundary & ~preserve] = WALL
    return labels

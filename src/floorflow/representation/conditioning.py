"""Exterior-contour conditioning features."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _shape_descriptor(footprint: torch.Tensor) -> torch.Tensor:
    """Return vectorized scale/shape moments without dynamic indexing."""

    mask = footprint[:, 0].float()
    _batch, height, width = mask.shape
    y = torch.linspace(-1.0, 1.0, height, device=mask.device, dtype=mask.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=mask.device, dtype=mask.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mass = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    center_x = (mask * xx[None]).flatten(1).sum(dim=1) / mass
    center_y = (mask * yy[None]).flatten(1).sum(dim=1) / mass
    dx = xx[None] - center_x[:, None, None]
    dy = yy[None] - center_y[:, None, None]
    second_x = (mask * dx.square()).flatten(1).sum(dim=1) / mass
    second_y = (mask * dy.square()).flatten(1).sum(dim=1) / mass
    second_xy = (mask * dx * dy).flatten(1).sum(dim=1) / mass
    disc = torch.sqrt(
        (second_x - second_y).square() + 4.0 * second_xy.square()
    ).clamp_min(1.0e-6)
    area_fraction = mass / float(max(1, height * width))
    anisotropy = disc / (second_x + second_y).clamp_min(1.0e-6)
    return torch.stack(
        [
            area_fraction,
            anisotropy,
            (second_x - second_y) / disc,
            2.0 * second_xy / disc,
        ],
        dim=1,
    )


def _iterative_depth(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    depth = torch.zeros_like(mask)
    current = mask
    for _ in range(int(iterations)):
        current = -F.max_pool2d(-current, 3, stride=1, padding=1)
        depth = depth + current
    return depth / float(max(1, int(iterations)))


def prepare_condition(
    footprint: torch.Tensor,
    *,
    depth_iterations: int = 12,
    fourier_conditioning: bool = True,
) -> torch.Tensor:
    """Build occupancy, boundary, signed depth, and coordinate channels."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    footprint = footprint.float()
    eroded = -F.max_pool2d(-footprint, 3, stride=1, padding=1)
    boundary = (footprint - eroded).clamp(0.0, 1.0)
    inside_depth = _iterative_depth(footprint, depth_iterations)
    outside_depth = _iterative_depth(1.0 - footprint, max(4, depth_iterations // 2))
    batch, _, height, width = footprint.shape
    y = torch.linspace(-1.0, 1.0, height, device=footprint.device, dtype=footprint.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=footprint.device, dtype=footprint.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([xx, yy], dim=0)[None].expand(batch, -1, -1, -1)
    channels = [footprint, boundary, inside_depth, outside_depth, coords]
    if fourier_conditioning:
        descriptor = _shape_descriptor(footprint).view(batch, 4, 1, 1).expand(
            -1,
            -1,
            height,
            width,
        )
        channels.append(descriptor)
    return torch.cat(channels, dim=1)

"""Differentiable polygon rasterization adapted from Raster2Seq (MIT).

The upstream implementation (``third_party/Raster2Seq/diff_ras/polygon.py``,
SIGGRAPH 2026) renders a polygon with a signed-distance field and an
even-odd inside test.  Its CUDA extension (``native_rasterizer``) is optional
here; the pure-PyTorch path below mirrors ``SoftPolygonPyTorch`` and is used
whenever the extension has not been built.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - depends on an optional CUDA build
    import native_rasterizer as _native_rasterizer

    _NATIVE_AVAILABLE = True
except Exception:  # noqa: BLE001  # pragma: no cover - optional CUDA build
    _native_rasterizer = None
    _NATIVE_AVAILABLE = False


def native_available() -> bool:
    """Return whether the optional Raster2Seq CUDA rasterizer is importable."""

    return _NATIVE_AVAILABLE


def _inside_mask(vertices: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Even-odd ray-casting mask; vertices are ``[N,P,2]`` in pixel units."""

    device = vertices.device
    dtype = vertices.dtype
    y = torch.arange(grid_size, device=device, dtype=dtype)
    x = torch.arange(grid_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    inside = torch.zeros(
        (vertices.shape[0], grid_size, grid_size),
        dtype=torch.bool,
        device=device,
    )
    previous = vertices.shape[1] - 1
    for index in range(vertices.shape[1]):
        from_x = vertices[:, index, 0][:, None, None]
        from_y = vertices[:, index, 1][:, None, None]
        to_x = vertices[:, previous, 0][:, None, None]
        to_y = vertices[:, previous, 1][:, None, None]
        crosses = (from_y > grid_y[None]) != (to_y > grid_y[None])
        denominator = torch.where(
            to_y - from_y == 0,
            torch.ones_like(to_y),
            to_y - from_y,
        )
        intersection_x = (
            (to_x - from_x) * (grid_y[None] - from_y) / denominator + from_x
        )
        left_of_edge = grid_x[None] < intersection_x
        inside ^= crosses & left_of_edge
        previous = index
    return inside


def _segment_squared_distance(
    vertices: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Minimum squared distance from each pixel to the polygon boundary."""

    device = vertices.device
    dtype = vertices.dtype
    y = torch.arange(grid_size, device=device, dtype=dtype)
    x = torch.arange(grid_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    points = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    starts = vertices
    ends = torch.roll(vertices, shifts=-1, dims=1)
    delta = ends - starts
    length_sq = (delta * delta).sum(dim=-1).clamp_min(1.0e-5)
    offsets = points[None, None] - starts[:, :, None]
    ratio = (offsets * delta[:, :, None]).sum(dim=-1) / length_sq[:, :, None]
    ratio = ratio.clamp(0.0, 1.0)
    projection = starts[:, :, None] + ratio[..., None] * delta[:, :, None]
    distance = ((points[None, None] - projection) ** 2).sum(dim=-1)
    return distance.min(dim=1).values.reshape(
        vertices.shape[0],
        grid_size,
        grid_size,
    )


def diff_soft_polygon_masks(
    polygons: torch.Tensor,
    grid_size: int,
    *,
    smoothness: float = 1.0,
    query_chunk: int = 32,
) -> torch.Tensor:
    """Render ``[B,Q,P,2]`` normalized polygons to ``[B,Q,H,W]`` soft masks."""

    squeeze = polygons.ndim == 3
    if squeeze:
        polygons = polygons[None]
    if polygons.ndim != 4 or polygons.shape[-1] != 2:
        raise ValueError("polygons must be [B,Q,P,2]")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    batch, queries, points, _ = polygons.shape
    scale = float(grid_size - 1)
    vertices = polygons.reshape(batch * queries, points, 2) * scale
    masks = []
    chunk = max(1, int(query_chunk))
    for start in range(0, vertices.shape[0], chunk):
        piece = vertices[start : start + chunk]
        inside = _inside_mask(piece, grid_size)
        sign = inside.to(piece.dtype) * 2.0 - 1.0
        distance = _segment_squared_distance(piece, grid_size)
        masks.append(torch.sigmoid(distance * sign / float(smoothness)))
    output = torch.cat(masks, dim=0)
    output = output.reshape(batch, queries, grid_size, grid_size)
    return output[0] if squeeze else output


__all__ = ["diff_soft_polygon_masks", "native_available"]

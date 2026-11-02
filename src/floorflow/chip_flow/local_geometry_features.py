"""Learned raw local-geometry features omitted by the legacy 19 columns."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.graph_conditioning import pin_local_statistics
from floorflow.chip_flow.sparse_layers import _edge_mask

LOCAL_GEOMETRY_DIM = 9


def raw_local_geometry_features(
    areas: torch.Tensor,
    b2b: torch.Tensor | None,
    p2b: torch.Tensor | None,
    pins: torch.Tensor | None,
    n_b2b: torch.Tensor | None,
    n_p2b: torch.Tensor | None,
    n_pins: torch.Tensor | None,
) -> torch.Tensor:
    """Build target-free edge-distribution and pin-local statistics."""

    batch, count = areas.shape
    output = areas.new_zeros(batch, count, LOCAL_GEOMETRY_DIM)
    if b2b is not None and b2b.shape[1]:
        source = b2b[..., 0].long()
        target = b2b[..., 1].long()
        weight = b2b[..., 2].float().clamp_min(0.0)
        valid = (
            _edge_mask(n_b2b, b2b)
            & (source >= 0)
            & (source < count)
            & (target >= 0)
            & (target < count)
        )
        offsets = torch.arange(batch, device=b2b.device)[:, None] * count
        left = (source + offsets).reshape(-1)
        right = (target + offsets).reshape(-1)
        valid_flat = valid.reshape(-1)
        weight_flat = weight.reshape(-1)
        edge_rows = torch.cat([left[valid_flat], right[valid_flat]])
        edge_values = torch.cat([weight_flat[valid_flat], weight_flat[valid_flat]])
        degree_weight = areas.new_zeros(batch * count)
        degree_weight.index_add_(0, edge_rows, edge_values)
        maximum = areas.new_zeros(batch * count)
        maximum.scatter_reduce_(0, edge_rows, edge_values, reduce="amax")
        degree_count = areas.new_zeros(batch * count)
        degree_count.index_add_(
            0,
            edge_rows,
            torch.ones_like(edge_values),
        )
        output[..., 0] = degree_weight.log1p().reshape(batch, count)
        output[..., 1] = maximum.log1p().reshape(batch, count)
        output[..., 2] = (
            degree_count.reshape(batch, count) / max(count, 1)
        )
    if p2b is not None and pins is not None and p2b.shape[1] and pins.shape[1]:
        pin_weight = p2b[..., 2].float().clamp_min(0.0)
        valid = _edge_mask(n_p2b, p2b)
        block = p2b[..., 1].long()
        valid = valid & block.ge(0) & block.lt(count)
        offsets = torch.arange(batch, device=p2b.device)[:, None] * count
        block_flat = (block + offsets).reshape(-1)
        valid_flat = valid.reshape(-1)
        weight_flat = pin_weight.reshape(-1)
        p2b_weight = areas.new_zeros(batch * count)
        p2b_weight.index_add_(
            0,
            block_flat[valid_flat],
            weight_flat[valid_flat],
        )
        p2b_max = areas.new_zeros(batch * count)
        p2b_max.scatter_reduce_(
            0,
            block_flat[valid_flat],
            weight_flat[valid_flat],
            reduce="amax",
        )
        output[..., 3] = p2b_weight.log1p().reshape(batch, count)
        output[..., 4] = p2b_max.log1p().reshape(batch, count)
        xy, strength, spread = pin_local_statistics(
            areas,
            p2b,
            pins,
            n_p2b,
            n_pins,
        )
        output[..., 5:7] = xy
        output[..., 7] = spread
        output[..., 8] = strength
    return output


class RawLocalGeometryEncoder(nn.Module):
    """Zero-gated learned encoding of raw local edge/pin statistics."""

    def __init__(self, hidden: int, *, width: int | None = None) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        bottleneck = int(width or max(16, hidden // 8))
        self.network = nn.Sequential(
            nn.LayerNorm(LOCAL_GEOMETRY_DIM),
            nn.Linear(LOCAL_GEOMETRY_DIM, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != LOCAL_GEOMETRY_DIM:
            raise ValueError("local geometry feature width mismatch")
        return self.gate * self.network(features.float())


__all__ = [
    "LOCAL_GEOMETRY_DIM",
    "RawLocalGeometryEncoder",
    "raw_local_geometry_features",
]

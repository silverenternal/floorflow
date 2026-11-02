"""Graph-conditioned inputs adapted from the project's pinned Trinity clone."""

from __future__ import annotations

import torch
from torch import nn


class GraphAttentionBias(nn.Module):
    """Learned additive bias from normalized b2b adjacency."""

    def __init__(self, init_gate: float = 1.0) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(
        self,
        adjacency: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bias = self.gate * adjacency
        if key_mask is not None:
            pad = torch.where(
                key_mask,
                torch.zeros((), device=key_mask.device, dtype=bias.dtype),
                torch.full((), -1.0e4, device=key_mask.device, dtype=bias.dtype),
            )
            bias = bias + pad[:, None, :]
        return bias


class GraphMessageResidual(nn.Module):
    """Zero-initialized normalized neighbor message residual."""

    def __init__(self, hidden: int, normalize: str = "row") -> None:
        super().__init__()
        if normalize not in {"row", "sym", "none"}:
            raise ValueError("normalize must be row, sym, or none")
        self.normalize = normalize
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def _normalized(self, adjacency: torch.Tensor) -> torch.Tensor:
        if self.normalize == "none":
            return adjacency
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        if self.normalize == "row":
            return adjacency / degree
        inverse = degree.rsqrt()
        return inverse * adjacency * inverse.transpose(-1, -2)

    def forward(
        self,
        tokens: torch.Tensor,
        adjacency: torch.Tensor | None,
    ) -> torch.Tensor:
        if adjacency is None:
            return tokens
        messages = torch.bmm(
            self._normalized(adjacency).to(tokens.dtype),
            self.norm(tokens),
        )
        return tokens + self.gate * self.proj(messages)


def dense_b2b_adjacency(
    b2b: torch.Tensor,
    n_b2b: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Build symmetric log1p b2b adjacency from padded edge batches."""

    batch, _, _ = b2b.shape
    adjacency = b2b.new_zeros(batch * count * count)
    rows = torch.arange(b2b.shape[1], device=b2b.device)[None, :]
    valid = rows < n_b2b.to(b2b.device).long()[:, None]
    source = b2b[..., 0].long().clamp(0, count - 1)
    target = b2b[..., 1].long().clamp(0, count - 1)
    weight = b2b[..., 2].clamp_min(0.0)
    batch_offset = torch.arange(batch, device=b2b.device)[:, None] * count * count
    source_flat = (source * count + target + batch_offset).reshape(-1)
    target_flat = (target * count + source + batch_offset).reshape(-1)
    weight_flat = weight.reshape(-1)
    valid_flat = valid.reshape(-1)
    adjacency.index_add_(
        0,
        source_flat[valid_flat],
        weight_flat[valid_flat],
    )
    adjacency.index_add_(
        0,
        target_flat[valid_flat],
        weight_flat[valid_flat],
    )
    adjacency = adjacency.reshape(batch, count, count).log1p()
    maximum = adjacency.amax(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
    return adjacency / maximum


def pin_local_statistics(
    areas: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
    n_p2b: torch.Tensor,
    n_pins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted pin centroid, normalized strength and local spread."""

    batch, count = areas.shape
    dtype = areas.dtype
    pin_scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    edge_rows = torch.arange(p2b.shape[1], device=p2b.device)[None, :]
    valid = edge_rows < n_p2b.to(p2b.device).long()[:, None]
    pin_index = p2b[..., 0].long()
    block_index = p2b[..., 1].long()
    valid = (
        valid
        & (pin_index >= 0)
        & (pin_index < n_pins.to(p2b.device).long()[:, None])
        & (block_index >= 0)
        & (block_index < count)
    )
    batch_offset = torch.arange(batch, device=p2b.device)[:, None]
    pin_flat = (pin_index + batch_offset * pins.shape[1]).reshape(-1)
    block_flat = (block_index + batch_offset * count).reshape(-1)
    weight_flat = p2b[..., 2].reshape(-1)
    valid_flat = valid.reshape(-1)
    flat_pins = pins.reshape(-1, pins.shape[-1])
    weighted = flat_pins.index_select(0, pin_flat[valid_flat])
    weighted = weighted * weight_flat[valid_flat, None]
    coordinate_sums = torch.zeros(
        batch * count,
        2,
        device=areas.device,
        dtype=dtype,
    )
    weight_sums = torch.zeros(
        batch * count,
        device=areas.device,
        dtype=dtype,
    )
    coordinate_sums.index_add_(0, block_flat[valid_flat], weighted)
    weight_sums.index_add_(
        0,
        block_flat[valid_flat],
        weight_flat[valid_flat],
    )
    coordinate_sums = coordinate_sums.reshape(batch, count, 2)
    weight = weight_sums.reshape(batch, count)
    xy = coordinate_sums / weight[..., None].clamp_min(1.0e-6)
    xy = xy / pin_scale[:, None, None]
    full_pins = flat_pins.index_select(0, pin_flat[valid_flat])
    pin_batch = (
        torch.arange(batch, device=p2b.device)[:, None]
        .expand(batch, p2b.shape[1])
        .reshape(-1)[valid_flat]
    )
    normalized_pins = full_pins / pin_scale[pin_batch, None]
    full_block = block_flat[valid_flat]
    pin_delta = normalized_pins - xy.reshape(-1, 2).index_select(
        0,
        full_block,
    )
    spread_sums = pin_delta.square().sum(dim=-1) * weight_flat[valid_flat]
    spread = torch.zeros(
        batch * count,
        device=areas.device,
        dtype=dtype,
    )
    spread.index_add_(0, full_block, spread_sums)
    spread = (
        spread / weight_sums.clamp_min(1.0e-6)
    ).sqrt().reshape(batch, count).clamp_max(4.0)
    strength = weight.log1p()
    strength = strength / strength.amax(dim=1, keepdim=True).clamp_min(1.0e-6)
    return xy, strength, spread


def pin_pull_features(
    areas: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
    n_p2b: torch.Tensor,
    n_pins: torch.Tensor,
) -> torch.Tensor:
    xy, strength, _spread = pin_local_statistics(
        areas,
        p2b,
        pins,
        n_p2b,
        n_pins,
    )
    return torch.cat([xy, strength[..., None]], dim=-1)


def graph_conditioned_features(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 19-column block features and normalized dense adjacency."""

    areas = batch["areas"].float()
    constraints = batch["constraints"].float()
    boxes = batch["boxes"].float()
    batch_size, count = areas.shape
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    adjacency = dense_b2b_adjacency(
        batch["b2b"],
        batch["n_b2b"],
        count,
    )
    degree = adjacency.sum(dim=-1)
    degree = degree / degree.amax(dim=1, keepdim=True).clamp_min(1.0e-6)
    boundary = constraints[..., 4].long()
    boundary_bits = torch.stack(
        [
            boundary & 1,
            boundary & 2,
            boundary & 4,
            boundary & 8,
        ],
        dim=-1,
    ).float().sign()
    centers = (boxes[..., :2] + boxes[..., 2:] * 0.5) / scale[:, None, None]
    rho = torch.log(
        boxes[..., 2].clamp_min(1.0e-6) / boxes[..., 3].clamp_min(1.0e-6)
    )
    fixed = constraints[..., 0].ne(0)
    preplaced = constraints[..., 1].ne(0)
    anchored = fixed | preplaced
    anchor_z = torch.stack(
        [
            torch.where(preplaced, centers[..., 0], 0.0),
            torch.where(preplaced, centers[..., 1], 0.0),
            torch.where(anchored, rho, 0.0),
        ],
        dim=-1,
    )
    anchor_mask = torch.stack(
        [
            preplaced.float(),
            preplaced.float(),
            anchored.float(),
        ],
        dim=-1,
    )
    pin_pull = pin_pull_features(
        areas,
        batch["p2b"],
        batch["pins"],
        batch["n_p2b"],
        batch["n_pins"],
    )
    features = torch.cat(
        [
            (areas.sqrt() / scale[:, None])[..., None],
            fixed.float()[..., None],
            preplaced.float()[..., None],
            degree[..., None],
            boundary_bits,
            constraints[..., 2:4].ne(0).float(),
            anchor_z,
            anchor_mask,
            pin_pull,
        ],
        dim=-1,
    )
    key_mask = torch.ones(
        batch_size,
        count,
        device=areas.device,
        dtype=torch.bool,
    )
    return features, adjacency, key_mask


__all__ = [
    "GraphAttentionBias",
    "GraphMessageResidual",
    "dense_b2b_adjacency",
    "graph_conditioned_features",
    "pin_local_statistics",
    "pin_pull_features",
]

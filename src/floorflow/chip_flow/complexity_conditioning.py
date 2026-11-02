"""Explicit target-free case-complexity features for graph conditioning."""

from __future__ import annotations

import torch
from torch import nn


def case_complexity_features(
    areas: torch.Tensor,
    constraints: torch.Tensor,
    b2b: torch.Tensor | None,
    p2b: torch.Tensor | None,
    pins: torch.Tensor | None,
    n_b2b: torch.Tensor | None,
    n_p2b: torch.Tensor | None,
    n_pins: torch.Tensor | None,
) -> torch.Tensor:
    """Return normalized count/density/weight features available before sampling."""

    batch, count = areas.shape
    device = areas.device
    edge_count = torch.zeros(batch, device=device, dtype=areas.dtype)
    if n_b2b is not None:
        edge_count = n_b2b.to(device).to(areas.dtype)
    elif b2b is not None:
        edge_count = areas.new_full((batch,), float(b2b.shape[1]))
    pin_edge_count = torch.zeros_like(edge_count)
    if n_p2b is not None:
        pin_edge_count = n_p2b.to(device).to(areas.dtype)
    elif p2b is not None:
        pin_edge_count = areas.new_full((batch,), float(p2b.shape[1]))
    pin_count = torch.zeros_like(edge_count)
    if n_pins is not None:
        pin_count = n_pins.to(device).to(areas.dtype)
    elif pins is not None:
        pin_count = areas.new_full((batch,), float(pins.shape[1]))
    area_mean = areas.mean(dim=1).clamp_min(1.0e-6)
    area_cv = areas.std(dim=1, correction=0) / area_mean
    constraint_density = constraints.ne(0).float().mean(dim=(1, 2))
    b2b_weight = _masked_edge_mean(b2b, n_b2b, batch, device, areas.dtype)
    p2b_weight = _masked_edge_mean(p2b, n_p2b, batch, device, areas.dtype)
    size_fraction = areas.new_full(
        (batch,),
        (count - 21.0) / 99.0,
    ).clamp(0.0, 1.0)
    return torch.stack(
        [
            size_fraction,
            (edge_count / count).clamp(0.0, 8.0) / 8.0,
            (pin_edge_count / count).clamp(0.0, 16.0) / 16.0,
            (pin_count / count).clamp(0.0, 16.0) / 16.0,
            constraint_density,
            area_cv.clamp(0.0, 4.0) / 4.0,
            (b2b_weight + p2b_weight).clamp(0.0, 16.0) / 16.0,
        ],
        dim=-1,
    )


def _masked_edge_mean(
    edges: torch.Tensor | None,
    lengths: torch.Tensor | None,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if edges is None or edges.shape[1] == 0:
        return torch.zeros(batch, device=device, dtype=dtype)
    rows = torch.arange(edges.shape[1], device=device)[None]
    valid = (
        rows < lengths.to(device).long()[:, None]
        if lengths is not None
        else torch.ones(batch, edges.shape[1], device=device, dtype=torch.bool)
    )
    weight = edges[..., 2].to(device=device, dtype=dtype).clamp_min(0.0)
    return (weight * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


class ComplexityConditioning(nn.Module):
    """Zero-initialized case complexity injection into the global token."""

    def __init__(self, hidden: int, feature_dim: int = 7) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_dim, hidden)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features.float())


__all__ = ["ComplexityConditioning", "case_complexity_features"]

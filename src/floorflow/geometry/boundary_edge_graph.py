"""Explicit edge-token encoding for a boundary polygon."""

from __future__ import annotations

import torch
from torch import nn


def boundary_edge_features(
    points: torch.Tensor,
    valid: torch.Tensor,
    entrance_points: torch.Tensor | None = None,
    entrance_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode consecutive boundary segments without flattening coordinates."""

    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("boundary points must be [B,P,2]")
    if valid.shape != points.shape[:2]:
        raise ValueError("boundary validity must be [B,P]")
    batch, maximum, _ = points.shape
    count = valid.sum(dim=1).clamp_min(1)
    indices = torch.arange(maximum, device=points.device)
    next_index = torch.remainder(indices[None] + 1, count[:, None])
    previous_index = torch.remainder(indices[None] - 1, count[:, None])
    following = torch.gather(
        points,
        1,
        next_index[..., None].expand(-1, -1, 2),
    )
    delta = following - points
    length = delta.norm(dim=-1)
    edge_valid = (indices[None] < count[:, None]) & (count[:, None] >= 3)
    perimeter = (
        length * edge_valid.to(length.dtype)
    ).sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    cumulative = torch.cumsum(
        length * edge_valid.to(length.dtype),
        dim=1,
    ) / perimeter
    direction = delta / length[..., None].clamp_min(1.0e-8)
    previous = torch.gather(
        direction,
        1,
        previous_index[..., None].expand(-1, -1, 2),
    )
    following_direction = torch.gather(
        direction,
        1,
        next_index[..., None].expand(-1, -1, 2),
    )
    sine = previous[..., 0] * direction[..., 1] - previous[..., 1] * direction[..., 0]
    cosine = (previous * direction).sum(dim=-1)
    midpoint = 0.5 * (points + following)
    phase = indices[None].to(points.dtype) / count[:, None].to(points.dtype)
    is_entrance = torch.zeros_like(phase)
    is_entrance[:, 0] = edge_valid[:, 0].to(phase.dtype)
    if entrance_points is not None and entrance_valid is not None:
        if entrance_points.shape != (batch, 2, 2):
            raise ValueError("entrance points must be [B,2,2]")
        if entrance_valid.shape != (batch,):
            raise ValueError("entrance validity must be [B]")
        entrance_midpoint = entrance_points.mean(dim=1)
        entrance_distance = (
            midpoint - entrance_midpoint[:, None]
        ).norm(dim=-1).masked_fill(~edge_valid, torch.inf)
        entrance_index = entrance_distance.argmin(dim=1)
        entrance_rows = torch.arange(batch, device=points.device)
        entrance_mask = entrance_valid & (count >= 3)
        is_entrance[entrance_rows[entrance_mask], entrance_index[entrance_mask]] = 1.0
        is_entrance[entrance_rows[entrance_mask], 0] = 0.0
    features = torch.cat(
        [
            points,
            following,
            midpoint,
            delta,
            length[..., None],
            direction,
            previous,
            following_direction,
            sine[..., None],
            cosine[..., None],
            cumulative[..., None],
            phase[..., None],
            is_entrance[..., None],
        ],
        dim=-1,
    )
    features = features.masked_fill(~edge_valid[..., None], 0.0)
    return features, edge_valid


class BoundaryEdgeGraphEncoder(nn.Module):
    """Project explicit boundary segments into transformer memory tokens."""

    def __init__(
        self,
        hidden_dim: int = 96,
        feature_dim: int = 20,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(int(feature_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            4,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        points: torch.Tensor,
        valid: torch.Tensor,
        entrance_points: torch.Tensor | None = None,
        entrance_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features, edge_valid = boundary_edge_features(
            points,
            valid,
            entrance_points,
            entrance_valid,
        )
        tokens = self.network(features) + self.edge_query.to(features.dtype)
        padding = ~edge_valid
        safe_padding = padding.clone()
        empty = padding.all(dim=1)
        safe_padding[empty, 0] = False
        attended = self.attention(
            self.norm(tokens),
            self.norm(tokens),
            self.norm(tokens),
            key_padding_mask=safe_padding,
            need_weights=False,
        )[0]
        tokens = tokens + attended * edge_valid[..., None].to(tokens.dtype)
        return {
            "tokens": tokens,
            "valid": edge_valid,
            "features": features,
        }


__all__ = ["BoundaryEdgeGraphEncoder", "boundary_edge_features"]

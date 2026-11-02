"""Local residual refinement for readout-derived vector primitives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _coordinate_features(points: torch.Tensor) -> torch.Tensor:
    angles = points * math.pi * 2.0
    return torch.cat(
        [
            points,
            torch.sin(angles),
            torch.cos(angles),
        ],
        dim=-1,
    )


class VectorProposalRefiner(nn.Module):
    """Refine existing vector proposals without changing primitive identity."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 96,
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Conv2d(in_channels, hidden_dim, 1)
        coordinate_dim = 6
        self.room_encoder = nn.Sequential(
            nn.Linear(hidden_dim + coordinate_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.wall_encoder = nn.Sequential(
            nn.Linear(hidden_dim + coordinate_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.opening_encoder = nn.Sequential(
            nn.Linear(hidden_dim + coordinate_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.graph_projection = nn.Linear(hidden_dim, hidden_dim)
        self.room_residual = nn.Linear(hidden_dim, 2)
        self.wall_residual = nn.Linear(hidden_dim, 2)
        self.opening_residual = nn.Linear(hidden_dim, 2)
        self.residual_scale = float(residual_scale)
        for head in (
            self.room_residual,
            self.wall_residual,
            self.opening_residual,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _sample(
        self,
        hidden: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        if points.ndim == 3:
            points = points[:, :, None]
        grid = points * 2.0 - 1.0
        sampled = F.grid_sample(
            hidden,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.permute(0, 2, 3, 1)

    def _encode(
        self,
        encoder: nn.Module,
        hidden: torch.Tensor,
        points: torch.Tensor,
        valid: torch.Tensor,
        graph_token: torch.Tensor | None,
    ) -> torch.Tensor:
        point_values = points[:, :, None] if points.ndim == 3 else points
        sampled = self._sample(hidden, points)
        values = torch.cat(
            [
                sampled,
                _coordinate_features(point_values),
                valid[..., None].to(sampled.dtype),
            ],
            dim=-1,
        )
        encoded = encoder(values)
        if graph_token is not None:
            encoded = encoded + self.graph_projection(graph_token)[:, None, None]
        return encoded

    def forward(
        self,
        features: torch.Tensor,
        room_polygons: torch.Tensor,
        room_valid: torch.Tensor,
        wall_endpoints: torch.Tensor,
        wall_valid: torch.Tensor,
        opening_centers: torch.Tensor,
        opening_valid: torch.Tensor,
        graph_token: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.feature_projection(features)
        room_hidden = self._encode(
            self.room_encoder,
            hidden,
            room_polygons,
            room_valid[..., None].expand_as(room_polygons[..., 0]),
            graph_token,
        )
        wall_hidden = self._encode(
            self.wall_encoder,
            hidden,
            wall_endpoints,
            wall_valid[..., None].expand_as(wall_endpoints[..., 0]),
            graph_token,
        )
        opening_hidden = self._encode(
            self.opening_encoder,
            hidden,
            opening_centers,
            opening_valid[..., None],
            graph_token,
        )
        room_residual = torch.tanh(self.room_residual(room_hidden))
        wall_residual = torch.tanh(self.wall_residual(wall_hidden))
        opening_residual = torch.tanh(
            self.opening_residual(opening_hidden)
        ).squeeze(2)
        room_residual = room_residual * room_valid[..., None, None]
        wall_residual = wall_residual * wall_valid[..., None, None]
        opening_residual = opening_residual * opening_valid[..., None]
        return {
            "room_polygons": (
                room_polygons + self.residual_scale * room_residual
            ).clamp(0.0, 1.0),
            "wall_endpoints": (
                wall_endpoints + self.residual_scale * wall_residual
            ).clamp(0.0, 1.0),
            "opening_centers": (
                opening_centers + self.residual_scale * opening_residual
            ).clamp(0.0, 1.0),
            "room_residual": room_residual,
            "wall_residual": wall_residual,
            "opening_residual": opening_residual,
        }


__all__ = ["VectorProposalRefiner"]

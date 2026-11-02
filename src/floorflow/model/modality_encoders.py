"""Token encoders for structured floorplan modalities."""

from __future__ import annotations

import torch
import torch.nn as nn


class FloorplanModalityEncoders(nn.Module):
    def __init__(self, hidden_dim: int, room_features: int, wall_features: int, opening_features: int, furniture_features: int) -> None:
        super().__init__()
        self.room = nn.Sequential(nn.Linear(room_features, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.wall = nn.Sequential(nn.Linear(wall_features, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.opening = nn.Sequential(nn.Linear(opening_features, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.furniture = nn.Sequential(nn.Linear(furniture_features, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        room_features: torch.Tensor,
        wall_features: torch.Tensor,
        opening_features: torch.Tensor,
        furniture_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "room": self.norm(self.room(room_features)),
            "wall": self.norm(self.wall(wall_features)),
            "opening": self.norm(self.opening(opening_features)),
            "furniture": self.norm(self.furniture(furniture_features)),
        }

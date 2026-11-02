"""Encode an explicit entrance interval as a first-class geometry token."""

from __future__ import annotations

import torch
from torch import nn


def entrance_geometry_features(
    points: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Build normalized start, end, direction, length, and validity features."""

    if points.ndim != 3 or points.shape[-2:] != (2, 2):
        raise ValueError("entrance points must be [B,2,2]")
    if valid.shape != points.shape[:1]:
        raise ValueError("entrance validity must be [B]")
    delta = points[:, 1] - points[:, 0]
    length = delta.norm(dim=-1, keepdim=True)
    direction = delta / length.clamp_min(1.0e-6)
    angle = torch.atan2(direction[..., 1:2], direction[..., 0:1])
    features = torch.cat(
        [
            points.flatten(start_dim=1),
            points.mean(dim=1),
            length,
            direction,
            torch.sin(angle),
            torch.cos(angle),
            valid[..., None].to(points.dtype),
        ],
        dim=-1,
    )
    return features.masked_fill(~valid[:, None], 0.0)


class EntranceGeometryEncoder(nn.Module):
    """Map one entrance interval into the shared boundary-token space."""

    def __init__(self, hidden_dim: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.validity_gate = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.network[2].weight)
        nn.init.zeros_(self.network[2].bias)
        nn.init.zeros_(self.validity_gate.weight)
        nn.init.zeros_(self.validity_gate.bias)

    def forward(
        self,
        points: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.network(entrance_geometry_features(points, valid))
        gate = torch.sigmoid(self.validity_gate(hidden))
        token = hidden * gate * valid[:, None].to(hidden.dtype)
        return {"tokens": token[:, None], "valid": valid.bool()}


__all__ = ["EntranceGeometryEncoder", "entrance_geometry_features"]

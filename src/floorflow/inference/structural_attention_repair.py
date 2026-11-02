"""Local structural attention repair for wall/opening/furniture constraints."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralAttentionRepair(nn.Module):
    def __init__(self, hidden_dim: int = 32, max_delta: float = 0.15) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.risk = nn.Sequential(
            nn.Conv2d(hidden_dim + 3, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.delta = nn.Sequential(
            nn.Conv2d(hidden_dim + 3, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        opening_risk: torch.Tensor,
        containment_risk: torch.Tensor,
        boundary_risk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        risks = torch.stack([opening_risk, containment_risk, boundary_risk], dim=1)
        joined = torch.cat([features, risks], dim=1)
        gate = torch.sigmoid(self.risk(joined))
        delta = torch.tanh(self.delta(joined)) * self.max_delta
        return delta * gate, gate

"""Risk-aware route mixer for floorplan coordinate updates."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayoutRouter(nn.Module):
    def __init__(self, hidden_dim: int, route_names: tuple[str, ...], risk_dim: int = 4) -> None:
        super().__init__()
        self.route_names = tuple(route_names)
        self.route_logits = nn.Sequential(
            nn.Linear(hidden_dim + len(self.route_names) * 4 + risk_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(self.route_names)),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.route_logits[-1].weight)
        nn.init.zeros_(self.route_logits[-1].bias)

    def forward(
        self,
        pooled_features: torch.Tensor,
        deltas: dict[str, torch.Tensor],
        risk: torch.Tensor,
        time_value: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        available = [name for name in self.route_names if name in deltas]
        if not available:
            raise ValueError("no active layout routes")
        norms = []
        for name in available:
            norm = deltas[name].flatten(1).norm(dim=1)
            norms.extend([norm, norm.pow(2), torch.log1p(norm), torch.ones_like(norm)])
        norms = torch.stack(norms, dim=1)
        features = torch.cat([pooled_features, norms, risk, time_value[:, None]], dim=1)
        logits = self.route_logits(features).gather(1, torch.tensor(
            [self.route_names.index(name) for name in available], device=features.device
        ).expand(features.shape[0], -1))
        weights = torch.softmax(logits / self.temperature.clamp(0.35, 2.5), dim=1)
        mixed = sum(weights[:, index, None, None] * deltas[name] for index, name in enumerate(available))
        route_map = {name: weights[:, index] for index, name in enumerate(available)}
        diagnostics = {
            "route_entropy": -(weights * weights.clamp_min(1.0e-8).log()).sum(dim=1).mean(),
            "route_balance": (weights.mean(dim=0) - 1.0 / len(available)).pow(2).mean(),
        }
        return mixed, route_map, diagnostics

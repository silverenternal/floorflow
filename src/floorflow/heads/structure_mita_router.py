"""Multi-scale sparse routing for residential CAD structure refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.model.official_mita_route import OfficialMiTARouteAttention


def structure_route_metrics(
    diagnostics: object,
) -> dict[str, float]:
    """Flatten structure-MIta diagnostics for validation reporting."""

    if not isinstance(diagnostics, dict):
        return {}
    values = {
        "structure_route_budget": diagnostics.get(
            "route_budget_soft_mean"
        ),
        "structure_route_entropy": diagnostics.get("routing_entropy"),
        "structure_route_attention_entropy": diagnostics.get(
            "attention_entropy"
        ),
        "structure_route_effective_expert_fraction": diagnostics.get(
            "effective_expert_fraction"
        ),
        "structure_route_gate_mean": diagnostics.get("route_gate_mean"),
    }
    return {
        name: float(value.detach().float().mean().cpu())
        for name, value in values.items()
        if isinstance(value, torch.Tensor)
        and torch.isfinite(value.detach().float().mean())
    }


class MultiScaleStructureRoute(nn.Module):
    """Route the structure bottleneck through room and wall scale evidence."""

    def __init__(
        self,
        low_channels: int,
        skip64_channels: int,
        skip128_channels: int,
        *,
        route_grid: int = 16,
        router_count: int = 16,
        top_k: int = 8,
        heads: int = 4,
    ) -> None:
        super().__init__()
        channels = int(low_channels)
        if channels % heads:
            heads = 1
        self.channels = channels
        self.route_grid = max(1, int(route_grid))
        self.router_count = max(1, int(router_count))
        self.top_k = max(1, int(top_k))
        self.skip64 = nn.Conv2d(int(skip64_channels), channels, 1)
        self.skip128 = nn.Conv2d(int(skip128_channels), channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.route = OfficialMiTARouteAttention(
            channels,
            num_heads=int(heads),
            router_count=self.router_count,
            top_k=self.top_k,
        )
        self.budget = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.SiLU(),
            nn.Linear(channels, self.router_count),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.01))
        nn.init.zeros_(self.budget[-1].weight)
        nn.init.zeros_(self.budget[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _pool(self, value: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            value,
            (self.route_grid, self.route_grid),
        )

    def forward(
        self,
        low: torch.Tensor,
        skip64: torch.Tensor,
        skip128: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        low_small = self._pool(low)
        skip64_small = self._pool(self.skip64(skip64))
        skip128_small = self._pool(self.skip128(skip128))
        query = low_small.flatten(2).transpose(1, 2)
        memory = self.norm(
            torch.cat(
                [
                    low_small.flatten(2).transpose(1, 2),
                    skip64_small.flatten(2).transpose(1, 2),
                    skip128_small.flatten(2).transpose(1, 2),
                ],
                dim=1,
            )
        )
        router_count = min(self.router_count, query.shape[1])
        base_top_k = min(self.top_k, memory.shape[1])
        budget_soft = 1.0 + torch.sigmoid(
            self.budget(
                torch.cat(
                    [
                        low_small.mean(dim=(2, 3)),
                        skip64_small.mean(dim=(2, 3)),
                        skip128_small.mean(dim=(2, 3)),
                    ],
                    dim=-1,
                )
            )
        )[..., :router_count] * (base_top_k - 1)
        budgets = budget_soft.detach().round().clamp(
            1,
            base_top_k,
        ).to(torch.long)
        routed, diagnostics = self.route(
            query,
            memory,
            router_budgets=budgets,
        )
        routed = routed.transpose(1, 2).reshape(
            low.shape[0],
            self.channels,
            self.route_grid,
            self.route_grid,
        )
        routed = F.interpolate(
            routed,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gate = torch.sigmoid(self.gate(torch.cat([low, routed], dim=1)))
        strength = budget_soft.mean(dim=1)[:, None, None, None] / base_top_k
        output = low + (
            torch.tanh(self.scale) * strength * gate * routed
        )
        return output, {
            **diagnostics,
            "route_budget_soft_mean": budget_soft.mean(),
            "route_gate_mean": gate.mean(),
        }


__all__ = ["MultiScaleStructureRoute", "structure_route_metrics"]

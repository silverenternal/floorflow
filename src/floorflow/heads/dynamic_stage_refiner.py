"""Dynamic stage-specific heads for weak predicted semantic channels."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.geometry.deterministic_pooling import fixed_grid_pool
from floorflow.heads.room_segmenter import ConvNormAct
from floorflow.model.official_mita_route import OfficialMiTARouteAttention


class DynamicStageRefiner(nn.Module):
    """Fuse field features, predecessor evidence, and a stage token."""

    def __init__(
        self,
        field_channels: int,
        prior_channels: int,
        stage_token_dim: int,
        *,
        hidden_channels: int = 96,
        conditioning_channels: int = 0,
        route_attention: bool = False,
        route_grid: int = 8,
        route_router_count: int = 12,
        route_top_k: int = 6,
        route_heads: int = 4,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        self.field_projection = ConvNormAct(
            int(field_channels),
            hidden,
        )
        self.prior_projection = (
            nn.Conv2d(int(prior_channels), hidden, 1)
            if prior_channels
            else None
        )
        self.condition_projection = (
            nn.Conv2d(int(conditioning_channels), hidden, 1)
            if conditioning_channels
            else None
        )
        self.film = nn.Linear(int(stage_token_dim), hidden * 2)
        self.local = nn.Sequential(
            ConvNormAct(hidden, hidden),
            ConvNormAct(hidden, hidden),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, 1),
        )
        self.token_gate = nn.Linear(int(stage_token_dim), 1)
        self.route_attention = bool(route_attention)
        self.route_grid = max(1, int(route_grid))
        self.route_router_count = max(1, int(route_router_count))
        self.route_top_k = max(1, int(route_top_k))
        if self.route_attention:
            route_heads = int(route_heads)
            if hidden % route_heads:
                route_heads = 1
            self.route = OfficialMiTARouteAttention(
                hidden,
                num_heads=route_heads,
                router_count=self.route_router_count,
                top_k=self.route_top_k,
            )
            self.route_memory_norm = nn.LayerNorm(hidden)
            self.route_token_projection = nn.Linear(
                int(stage_token_dim),
                hidden,
            )
            self.route_output = nn.Conv2d(hidden, hidden, 1)
            self.route_gate = nn.Sequential(
                ConvNormAct(hidden * 2, hidden),
                nn.Conv2d(hidden, 1, 1),
            )
            self.route_budget = nn.Sequential(
                nn.Linear(hidden + int(stage_token_dim), hidden // 2),
                nn.SiLU(),
                nn.Linear(hidden // 2, self.route_router_count),
            )
            self.route_scale = nn.Parameter(torch.tensor(0.01))
            nn.init.normal_(
                self.route_budget[-1].weight,
                mean=0.0,
                std=1.0e-3,
            )
            nn.init.zeros_(self.route_budget[-1].bias)
        else:
            self.route = None
            self.route_memory_norm = None
            self.route_token_projection = None
            self.route_output = None
            self.route_gate = None
            self.route_budget = None
            self.route_scale = None
        self.output = nn.Conv2d(hidden, int(prior_channels), 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.normal_(self.token_gate.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.token_gate.bias)

    def forward(
        self,
        features: torch.Tensor,
        prior: torch.Tensor,
        stage_token: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.field_projection(features.float())
        if self.prior_projection is not None:
            hidden = hidden + self.prior_projection(prior.float())
        if self.condition_projection is not None and condition is not None:
            hidden = hidden + self.condition_projection(
                condition.float()
            )
        scale, bias = self.film(stage_token.float()).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale[..., None, None]) + bias[
            ..., None, None
        ]
        hidden = hidden + self.local(hidden)
        route_diagnostics = None
        if (
            self.route is not None
            and self.route_output is not None
            and self.route_gate is not None
            and self.route_budget is not None
            and self.route_memory_norm is not None
            and self.route_token_projection is not None
            and self.route_scale is not None
        ):
            pooled = fixed_grid_pool(
                hidden,
                (self.route_grid, self.route_grid),
            )
            memory_parts = [pooled]
            if self.prior_projection is not None:
                memory_parts.append(
                    fixed_grid_pool(
                        self.prior_projection(prior.float()),
                        (self.route_grid, self.route_grid),
                    )
                )
            if self.condition_projection is not None and condition is not None:
                memory_parts.append(
                    fixed_grid_pool(
                        self.condition_projection(condition.float()),
                        (self.route_grid, self.route_grid),
                    )
                )
            query_tokens = pooled.flatten(2).transpose(1, 2)
            memory = torch.cat(
                [
                    part.flatten(2).transpose(1, 2)
                    for part in memory_parts
                ]
                + [
                    self.route_token_projection(
                        stage_token.float()
                    )[:, None]
                ],
                dim=1,
            )
            memory = self.route_memory_norm(memory)
            base_top_k = min(self.route_top_k, memory.shape[1])
            router_count = min(
                self.route_router_count,
                query_tokens.shape[1],
            )
            budget_soft = 1.0 + torch.sigmoid(
                self.route_budget(
                    torch.cat(
                        [
                            pooled.mean(dim=(2, 3)),
                            stage_token.float(),
                        ],
                        dim=-1,
                    )
                )
            )[..., :router_count] * (base_top_k - 1)
            budgets = budget_soft.detach().round().clamp(
                1,
                base_top_k,
            ).to(torch.long)
            routed, route_diagnostics = self.route(
                query_tokens,
                memory,
                router_budgets=budgets,
            )
            routed_map = routed.transpose(1, 2).reshape(
                hidden.shape[0],
                hidden.shape[1],
                self.route_grid,
                self.route_grid,
            )
            routed_map = nn.functional.interpolate(
                routed_map,
                size=hidden.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            gate = torch.sigmoid(
                self.route_gate(torch.cat([hidden, routed_map], dim=1))
            )
            budget_strength = (
                budget_soft.mean(dim=1)[:, None, None, None] / base_top_k
            )
            hidden = hidden + (
                torch.tanh(self.route_scale)
                * budget_strength
                * gate
                * self.route_output(routed_map)
            )
            route_diagnostics = {
                **route_diagnostics,
                "route_gate_mean": gate.mean(),
                "route_budget_soft_mean": budget_soft.mean(),
            }
        spatial_gate = torch.sigmoid(
            self.gate(hidden) + self.token_gate(stage_token.float())[
                ..., None, None
            ]
        )
        residual = self.output(hidden)
        return {
            "residual": residual,
            "spatial_gate": spatial_gate[:, 0],
            "route_diagnostics": route_diagnostics,
        }


__all__ = ["DynamicStageRefiner"]

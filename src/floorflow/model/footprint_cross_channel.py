"""Learned retrieval-free footprint prior routed across semantic families."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.model.official_mita_route import OfficialMiTARouteAttention


class FootprintCrossChannelPrior(nn.Module):
    """Route footprint evidence into room, wall, opening and furniture tokens."""

    GROUP_ORDER = ("wall", "opening", "room", "furniture", "support")

    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int,
        queries_per_group: Mapping[str, int],
        *,
        token_grid: int = 16,
        heads: int = 4,
        top_k: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_channels <= 0 or hidden_channels <= 0:
            raise ValueError("feature and hidden channels must be positive")
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        self.hidden_channels = int(hidden_channels)
        self.token_grid = int(token_grid)
        self.query_counts = {
            name: int(queries_per_group.get(name, 1))
            for name in self.GROUP_ORDER
        }
        self.feature_projection = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )
        self.footprint_projection = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )
        self.boundary_projection = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )
        self.boundary_attention = nn.MultiheadAttention(
            hidden_channels,
            heads,
            batch_first=True,
        )
        self.boundary_scale = nn.Parameter(torch.zeros(1))
        self.queries = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.randn(count, hidden_channels) * 0.02
                )
                for name, count in self.query_counts.items()
            }
        )
        self.routes = nn.ModuleDict(
            {
                name: OfficialMiTARouteAttention(
                    hidden_channels,
                    num_heads=heads,
                    router_count=min(16, max(1, count)),
                    top_k=min(top_k, token_grid * token_grid),
                    dropout=dropout,
                )
                for name, count in self.query_counts.items()
            }
        )
        self.norms = nn.ModuleDict(
            {name: nn.LayerNorm(hidden_channels) for name in self.GROUP_ORDER}
        )
        self.residual_scales = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in self.GROUP_ORDER}
        )
        self.cross_norm = nn.LayerNorm(hidden_channels)

    @staticmethod
    def _parent_groups(name: str) -> tuple[str, ...]:
        return {
            "wall": (),
            "opening": ("wall",),
            "room": ("wall", "opening"),
            "furniture": ("room", "wall"),
            "support": ("room",),
        }[name]

    @staticmethod
    def _boundary_mask(footprint: torch.Tensor) -> torch.Tensor:
        eroded = F.max_pool2d(
            footprint.float(),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return (footprint.float() - eroded).clamp(0.0, 1.0)

    def forward(
        self,
        features: torch.Tensor,
        footprint: torch.Tensor,
    ) -> dict[str, object]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        if footprint.shape[-2:] != features.shape[-2:]:
            footprint = F.interpolate(
                footprint.float(),
                size=features.shape[-2:],
                mode="nearest",
            )
        feature_memory = F.adaptive_avg_pool2d(
            self.feature_projection(features.float()),
            (self.token_grid, self.token_grid),
        ).flatten(2).transpose(1, 2)
        footprint_memory = F.adaptive_avg_pool2d(
            self.footprint_projection(footprint.float()),
            (self.token_grid, self.token_grid),
        ).flatten(2).transpose(1, 2)
        boundary_memory = F.adaptive_avg_pool2d(
            self.boundary_projection(
                self._boundary_mask(footprint.float())
            ),
            (self.token_grid, self.token_grid),
        ).flatten(2).transpose(1, 2)
        base_memory = torch.cat(
            [footprint_memory, feature_memory],
            dim=1,
        )
        tokens: dict[str, torch.Tensor] = {}
        diagnostics = {}
        boundary_scale = self.boundary_scale.to(base_memory.dtype)
        for name in self.GROUP_ORDER:
            parents = [
                tokens[parent][:, : self.query_counts[name]]
                for parent in self._parent_groups(name)
                if parent in tokens
            ]
            memory = torch.cat([base_memory, *parents], dim=1)
            query = self.queries[name].to(memory.dtype)[None].expand(
                features.shape[0],
                -1,
                -1,
            )
            routed, route_diagnostics = self.routes[name](query, memory)
            boundary_residual = self.boundary_attention(
                query,
                boundary_memory,
                boundary_memory,
                need_weights=False,
            )[0]
            refined = self.norms[name](
                query + routed + boundary_scale * boundary_residual
            )
            scale = self.residual_scales[name].to(refined.dtype)
            tokens[name] = scale * refined
            diagnostics[name] = {
                "routing_entropy": route_diagnostics["routing_entropy"],
                "attention_entropy": route_diagnostics["attention_entropy"],
                "effective_expert_fraction": route_diagnostics[
                    "effective_expert_fraction"
                ],
                "boundary_residual_scale": boundary_scale.detach().clone(),
            }
        return {"tokens": tokens, "diagnostics": diagnostics}


__all__ = ["FootprintCrossChannelPrior"]

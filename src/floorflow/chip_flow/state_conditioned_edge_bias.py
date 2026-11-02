"""Current-geometry modulation for the existing sparse netlist graph."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn


class StateConditionedEdgeBias(nn.Module):
    """Predict a bounded, zero-initialized bias on active b2b edges."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        feature_dim = 7
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        block_state: torch.Tensor,
        static_adjacency: torch.Tensor,
        cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if static_adjacency is None:
            raise ValueError("state-conditioned edges require static adjacency")
        source = cache["b2b_source"]
        target = cache["b2b_target"]
        if source.numel() == 0:
            return static_adjacency
        batch, count, _ = block_state.shape
        if static_adjacency.shape != (batch, count, count):
            raise ValueError("static adjacency shape must match block state")
        flat_state = block_state.float().reshape(batch * count, -1)
        if flat_state.shape[-1] != 3:
            raise ValueError("state-conditioned edges require 3D block state")
        source_state = flat_state.index_select(0, source)
        target_state = flat_state.index_select(0, target)
        source_area = cache["areas"].float().reshape(-1).index_select(0, source)
        target_area = cache["areas"].float().reshape(-1).index_select(0, target)
        delta = target_state - source_state
        distance = delta[..., :2].abs().sum(dim=-1, keepdim=True)
        area_ratio = (target_area / source_area.clamp_min(1.0e-6)).log()
        features = torch.cat(
            [
                delta[..., :2],
                delta[..., :2].abs(),
                distance,
                area_ratio.clamp(-4.0, 4.0)[:, None],
                cache["b2b_weight"].float().log1p()[:, None],
            ],
            dim=-1,
        )
        score = torch.tanh(self.network(features).squeeze(-1)).to(
            static_adjacency.dtype
        )
        dense = static_adjacency.new_zeros(batch * count * count)
        degree = static_adjacency.new_zeros(batch * count)
        source_local = source % count
        target_local = target % count
        dense.index_add_(0, source * count + target_local, score)
        dense.index_add_(0, target * count + source_local, score)
        degree.index_add_(0, source, torch.ones_like(score))
        degree.index_add_(0, target, torch.ones_like(score))
        dynamic = dense.reshape(batch, count, count)
        dynamic = dynamic / degree.reshape(batch, count, 1).clamp_min(1.0)
        return static_adjacency + dynamic


def condition_state_edges(
    condition,
    module: StateConditionedEdgeBias,
    block_state: torch.Tensor,
    cache: dict[str, torch.Tensor],
):
    """Replace only the graph tensor reused by every backbone block."""

    adjacency = module(block_state, condition.adjacency, cache)
    return replace(condition, adjacency=adjacency)


__all__ = ["StateConditionedEdgeBias", "condition_state_edges"]

"""Zero-initialized output head for hard-projector displacement."""

from __future__ import annotations

import torch
from torch import nn


class ProjectorResidualHead(nn.Module):
    """Predict a per-block legalization residual from tokens and raw x0."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        bottleneck = max(32, hidden // 4)
        self.network = nn.Sequential(
            nn.LayerNorm(hidden * 2 + 3),
            nn.Linear(hidden * 2 + 3, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        time_token: torch.Tensor,
    ) -> torch.Tensor:
        context = time_token[:, None].expand(-1, block_state.shape[1], -1)
        return self.network(
            torch.cat([block_tokens.float(), block_state.float(), context], dim=-1)
        )


__all__ = ["ProjectorResidualHead"]

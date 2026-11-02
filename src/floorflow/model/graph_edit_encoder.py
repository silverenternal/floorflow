"""Graph-edit conditioning for local floorplan modification."""

from __future__ import annotations

import torch
from torch import nn


class GraphEditEncoder(nn.Module):
    """Encode edited room nodes, adjacency, and semantic family context."""

    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 96,
        operations: int = 8,
        families: int = 5,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.operation = nn.Embedding(operations, hidden_dim)
        self.family = nn.Embedding(families, hidden_dim)
        self.message = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(int(layers))
            ]
        )
        self.norm = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(int(layers))]
        )
        self.global_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        family_ids: torch.Tensor,
        operation_ids: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [B,N,F]")
        batch, nodes, _features = node_features.shape
        if adjacency.shape != (batch, nodes, nodes):
            raise ValueError("adjacency must have shape [B,N,N]")
        if family_ids.shape != (batch, nodes):
            raise ValueError("family_ids must have shape [B,N]")
        if node_mask is None:
            node_mask = torch.ones(
                batch,
                nodes,
                dtype=torch.bool,
                device=node_features.device,
            )
        weights = adjacency.float().clamp_min(0.0)
        weights = weights * node_mask[:, None, :].float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        hidden = (
            self.node_projection(node_features)
            + self.family(family_ids.long())
            + self.operation(operation_ids.long())[:, None]
        )
        hidden = hidden * node_mask[..., None].to(hidden.dtype)
        for message, norm in zip(self.message, self.norm, strict=True):
            neighbors = torch.bmm(weights, hidden)
            hidden = norm(
                hidden
                + message(torch.cat([hidden, neighbors], dim=-1))
            )
            hidden = hidden * node_mask[..., None].to(hidden.dtype)
        denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = hidden.sum(dim=1) / denom.to(hidden.dtype)
        operation = self.operation(operation_ids.long())
        global_context = self.global_projection(
            torch.cat([pooled, operation], dim=-1)
        )
        return {
            "node_tokens": hidden * self.output_scale,
            "global_token": global_context * self.output_scale,
            "node_mask": node_mask,
        }


__all__ = ["GraphEditEncoder"]

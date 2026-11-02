"""Spatial experts for wall and opening heatmap residuals."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.representation.gate_diagnostics import stream_gate_diagnostics


class SpatialExpertMoE(nn.Module):
    """Top-k routed convolution experts with load diagnostics."""

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if experts <= 1:
            raise ValueError("experts must be greater than one")
        if not 1 <= top_k <= experts:
            raise ValueError("top_k must be in [1, experts]")
        self.experts = int(experts)
        self.top_k = int(top_k)
        self.router = nn.Conv2d(hidden_dim, self.experts, 1)
        self.bank = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, out_channels, 1),
                )
                for _ in range(self.experts)
            ]
        )
        for expert in self.bank:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        routing_logits = self.router(hidden)
        routing = routing_logits.softmax(dim=1)
        top_values, top_indices = routing.topk(self.top_k, dim=1)
        sparse = torch.zeros_like(routing).scatter(1, top_indices, top_values)
        sparse = sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        expert_outputs = torch.stack([expert(hidden) for expert in self.bank], dim=1)
        weights = sparse.unsqueeze(2)
        output = (weights * expert_outputs).sum(dim=1)
        token_gates = sparse.flatten(2).transpose(1, 2)
        diagnostics = stream_gate_diagnostics(token_gates)
        diagnostics["expert_usage"] = sparse.mean(dim=(2, 3))
        return output, diagnostics


__all__ = ["SpatialExpertMoE"]

"""Learned coarse-to-fine fusion between semantic anchors and flow output."""

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticAnchorFusion(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        hidden = max(16, int(hidden_channels))
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 2.0)

    def forward(
        self,
        flow_endpoint: torch.Tensor,
        anchor_logits: torch.Tensor,
        time_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_probability = anchor_logits.sigmoid()
        flow_probability = flow_endpoint.sigmoid()
        time_map = time_value[:, None, None, None].expand(
            -1,
            1,
            flow_endpoint.shape[-2],
            flow_endpoint.shape[-1],
        )
        gate = torch.sigmoid(
            self.net(torch.cat([flow_probability, anchor_probability, time_map], dim=1))
        )
        fused_probability = gate * anchor_probability + (1.0 - gate) * flow_probability
        fused_logits = torch.logit(fused_probability.clamp(1.0e-5, 1.0 - 1.0e-5))
        return fused_logits, gate

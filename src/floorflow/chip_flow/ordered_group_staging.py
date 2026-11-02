"""Order-conditioned group staging for the chip hierarchy flow."""

from __future__ import annotations

import math

import torch
from torch import nn


class OrderedGroupStaging(nn.Module):
    """Inject cluster-ordered, time-phased residuals into block trajectories."""

    def __init__(
        self,
        hidden: int,
        *,
        cluster_slots: int = 4,
        temperature: float = 0.5,
        phase_width: float = 0.3,
    ) -> None:
        super().__init__()
        if cluster_slots <= 0:
            raise ValueError("cluster_slots must be positive")
        if temperature <= 0.0 or phase_width <= 0.0:
            raise ValueError("temperature and phase_width must be positive")
        bottleneck = max(16, hidden // 4)
        self.cluster_slots = int(cluster_slots)
        self.temperature = float(temperature)
        self.phase_width = float(phase_width)
        self.group_norm = nn.LayerNorm(hidden)
        self.priority = nn.Linear(hidden, 1)
        self.state_encoder = nn.Linear(3, hidden)
        self.phase_encoder = nn.Sequential(
            nn.Linear(4, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 4, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 3),
        )
        nn.init.normal_(self.fuse[-1].weight, std=0.02)
        nn.init.zeros_(self.fuse[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def group_phase_gates(
        self,
        cluster_tokens: torch.Tensor,
        cluster_mask: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.group_norm(cluster_tokens)
        priority = self.priority(normalized).squeeze(-1)
        active = cluster_mask.to(dtype=cluster_tokens.dtype)
        pair = torch.sigmoid(
            (
                priority[:, :, None] - priority[:, None, :]
            )
            / self.temperature
        )
        rank = (pair * active[:, None, :]).sum(dim=-1) - 0.5
        count = active.sum(dim=-1)
        denominator = (count - 1.0).clamp_min(1.0)
        center = rank / denominator[:, None]
        center = torch.where(
            count[:, None] > 1.0,
            center,
            torch.full_like(center, 0.5),
        ).clamp(0.0, 1.0)
        phase_gate = torch.exp(
            -((time[:, None] - center) / self.phase_width).square()
        )
        phase_gate = phase_gate * active
        return priority, center, phase_gate

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        cluster_tokens: torch.Tensor,
        cluster_ids: torch.Tensor,
        cluster_mask: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, count, hidden = block_tokens.shape
        if cluster_tokens.shape != (batch, self.cluster_slots, hidden):
            raise ValueError("cluster token shape mismatch")
        if cluster_ids.shape != (batch, count):
            raise ValueError("cluster id shape mismatch")
        if cluster_mask.shape != (batch, self.cluster_slots):
            raise ValueError("cluster mask shape mismatch")
        if time.shape != (batch,):
            raise ValueError("time must have shape (batch,)")
        priority, center, phase_gate = self.group_phase_gates(
            cluster_tokens,
            cluster_mask,
            time.float(),
        )
        phase_features = torch.stack(
            [
                center,
                phase_gate,
                torch.sin(2.0 * math.pi * center),
                torch.cos(2.0 * math.pi * center),
            ],
            dim=-1,
        )
        phase_tokens = self.phase_encoder(phase_features)
        valid_block = (cluster_ids > 0) & (
            cluster_ids <= self.cluster_slots
        )
        gather_index = (cluster_ids - 1).clamp(
            0,
            self.cluster_slots - 1,
        )
        group_tokens = cluster_tokens.gather(
            1,
            gather_index[..., None].expand(-1, -1, hidden),
        )
        sampled_phase = phase_tokens.gather(
            1,
            gather_index[..., None].expand(-1, -1, hidden),
        )
        state_tokens = self.state_encoder(block_state.to(block_tokens))
        residual = self.fuse(
            torch.cat(
                [
                    block_tokens,
                    group_tokens,
                    self._time_token(time, hidden)
                    .to(block_tokens)
                    .expand_as(block_tokens),
                    sampled_phase + state_tokens,
                ],
                dim=-1,
            )
        )
        residual = residual * valid_block[..., None]
        return self.gate * residual, {
            "order_group_priority": priority,
            "order_group_phase_center": center,
            "order_group_phase_gate": phase_gate,
            "order_block_mask": valid_block,
            "order_fuse_norm": residual.detach().norm(dim=-1),
        }

    def _time_token(self, time: torch.Tensor, hidden: int) -> torch.Tensor:
        base = time.float()[:, None]
        frequency = torch.linspace(
            1.0,
            8.0,
            hidden // 2,
            device=time.device,
            dtype=torch.float32,
        )
        phase = base * frequency[None]
        encoded = torch.cat([phase.sin(), phase.cos()], dim=-1)
        if encoded.shape[-1] < hidden:
            encoded = torch.nn.functional.pad(
                encoded,
                (0, hidden - encoded.shape[-1]),
            )
        return encoded[:, None].to(dtype=torch.float32)


__all__ = ["OrderedGroupStaging"]

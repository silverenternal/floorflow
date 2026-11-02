"""Degree-conditioned relational expert for chip block tokens."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.sparse_layers import _edge_mask


class DegreeConditionedExpert(nn.Module):
    """Apply a low-rank relational expert to high-degree/high-pin blocks."""

    def __init__(
        self,
        hidden: int,
        *,
        bottleneck: int | None = None,
    ) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        width = int(bottleneck or max(16, hidden // 4))
        self.hidden = int(hidden)
        self.bottleneck = width
        self.route_norm = nn.LayerNorm(hidden + 2)
        self.route = nn.Sequential(
            nn.Linear(hidden + 2, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        self.expert_norm = nn.LayerNorm(hidden)
        self.expert = nn.Sequential(
            nn.Linear(hidden, width),
            nn.SiLU(),
            nn.Linear(width, hidden),
        )
        nn.init.normal_(self.expert[-1].weight, std=0.02)
        nn.init.zeros_(self.expert[-1].bias)
        self.scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _degrees(
        block_count: int,
        b2b: torch.Tensor | None,
        p2b: torch.Tensor | None,
        n_b2b: torch.Tensor | None,
        n_p2b: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = (n_b2b if n_b2b is not None else n_p2b).shape[0]
        b2b_degree = torch.zeros(batch, block_count, device=device, dtype=dtype)
        p2b_degree = torch.zeros_like(b2b_degree)
        if b2b is not None and b2b.shape[1]:
            valid = (
                _edge_mask(n_b2b, b2b)
                & (b2b[..., 0] >= 0)
                & (b2b[..., 0] < block_count)
                & (b2b[..., 1] >= 0)
                & (b2b[..., 1] < block_count)
            )
            weight = b2b[..., 2].to(dtype=dtype).clamp_min(0.0)
            batch_index = torch.arange(batch, device=device)[:, None]
            for column in (0, 1):
                flat_index = (
                    batch_index * block_count + b2b[..., column].long()
                )[valid]
                b2b_degree.view(-1).index_add_(
                    0,
                    flat_index,
                    weight[valid],
                )
        if p2b is not None and p2b.shape[1]:
            valid = (
                _edge_mask(n_p2b, p2b)
                & (p2b[..., 1] >= 0)
                & (p2b[..., 1] < block_count)
            )
            batch_index = torch.arange(batch, device=device)[:, None]
            flat_index = (
                batch_index * block_count + p2b[..., 1].long()
            )[valid]
            p2b_degree.view(-1).index_add_(
                0,
                flat_index,
                p2b[..., 2].to(dtype=dtype).clamp_min(0.0)[valid],
            )
        b2b_degree = b2b_degree.log1p()
        p2b_degree = p2b_degree.log1p()
        b2b_degree = b2b_degree / b2b_degree.amax(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        p2b_degree = p2b_degree / p2b_degree.amax(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        return b2b_degree, p2b_degree

    def forward(
        self,
        block_tokens: torch.Tensor,
        *,
        b2b: torch.Tensor | None,
        p2b: torch.Tensor | None,
        n_b2b: torch.Tensor | None,
        n_p2b: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, count, hidden = block_tokens.shape
        if hidden != self.hidden:
            raise ValueError("block token hidden size mismatch")
        if n_b2b is None and n_p2b is None:
            raise ValueError("degree expert requires at least one edge count")
        b2b_degree, p2b_degree = self._degrees(
            count,
            b2b,
            p2b,
            n_b2b,
            n_p2b,
            block_tokens.device,
            block_tokens.dtype,
        )
        route_features = torch.cat(
            [
                block_tokens,
                b2b_degree[..., None],
                p2b_degree[..., None],
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.route(self.route_norm(route_features)))
        expert = self.expert(self.expert_norm(block_tokens))
        residual = self.scale * gate * expert
        return block_tokens + residual, {
            "degree_expert_gate": gate.squeeze(-1),
            "degree_expert_b2b_degree": b2b_degree,
            "degree_expert_p2b_degree": p2b_degree,
        }


__all__ = ["DegreeConditionedExpert"]

"""Top-down hierarchy feedback for per-block chip flow trajectories."""

from __future__ import annotations

import torch
from torch import nn


class CoarseToFineCoupler(nn.Module):
    """Feed predicted L0/L1 states back into the L2 block trajectory.

    The existing hierarchy predicts global, group and block states in parallel.
    This coupler makes that prediction ordering explicit: the predicted boundary
    and structural states become a shared coarse field that corrects blocks at
    the current flow time. It uses only model predictions and input membership;
    target boxes and target classes are never accessed.
    """

    def __init__(
        self,
        hidden: int,
        *,
        structural_slots: int = 9,
        bottleneck: int | None = None,
    ) -> None:
        super().__init__()
        if hidden <= 0 or structural_slots <= 0:
            raise ValueError("hidden and structural_slots must be positive")
        width = int(bottleneck or max(16, hidden // 4))
        self.hidden = int(hidden)
        self.structural_slots = int(structural_slots)
        self.bottleneck = width
        self.global_encoder = nn.Sequential(
            nn.Linear(4, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.structural_encoder = nn.Linear(3, width)
        self.structural_type = nn.Parameter(
            torch.randn(self.structural_slots, width) * 0.02
        )
        self.state_encoder = nn.Linear(3, width)
        self.time_encoder = nn.Linear(1, width)
        self.reduce = nn.Sequential(
            nn.Linear(hidden + 4 * width, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )
        nn.init.normal_(self.reduce[-1].weight, std=0.02)
        nn.init.zeros_(self.reduce[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        membership: torch.Tensor,
        structural_mask: torch.Tensor,
        global_prediction: torch.Tensor,
        structural_prediction: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, count, hidden = block_tokens.shape
        if hidden != self.hidden:
            raise ValueError("block token hidden size mismatch")
        if block_state.shape != (batch, count, 3):
            raise ValueError("block state must have shape (batch, n, 3)")
        if membership.shape != (batch, count, self.structural_slots):
            raise ValueError("membership has the wrong structural-slot count")
        if structural_mask.shape != (batch, self.structural_slots):
            raise ValueError("structural mask must have shape (batch, slots)")
        if global_prediction.shape != (batch, 4):
            raise ValueError("global prediction must have shape (batch, 4)")
        if structural_prediction.shape != (
            batch,
            self.structural_slots,
            3,
        ):
            raise ValueError("structural prediction has the wrong shape")
        if time.shape != (batch,):
            raise ValueError("time must have shape (batch,)")

        weights = (
            membership.to(block_tokens.dtype)
            * structural_mask[:, None, :].to(block_tokens.dtype)
        )
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled_state = torch.bmm(weights, structural_prediction.float())
        pooled_state = pooled_state / denominator
        structural_code = self.structural_encoder(
            pooled_state.to(block_tokens.dtype)
        )
        active_types = torch.bmm(
            weights,
            self.structural_type[None].expand(batch, -1, -1),
        )
        active_types = active_types / denominator
        time_code = self.time_encoder(time.float()[:, None])
        context = torch.cat(
            [
                self.global_encoder(global_prediction.float())[:, None].expand(
                    -1,
                    count,
                    -1,
                ),
                structural_code + active_types,
                self.state_encoder(block_state.float()),
                time_code[:, None].expand(-1, count, -1),
            ],
            dim=-1,
        )
        residual = self.reduce(
            torch.cat(
                [block_tokens, context.to(block_tokens.dtype)],
                dim=-1,
            )
        )
        residual = self.gate * residual
        return residual, {
            "coarse_to_fine_active_fraction": denominator.gt(0).float().mean(),
            "coarse_to_fine_residual_norm": residual.detach().norm(dim=-1),
        }


__all__ = ["CoarseToFineCoupler"]

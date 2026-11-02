"""Complexity-conditioned sparse slot feedback for chip trajectories."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.sparse_hierarchy import LearnedSparseSlotRouter


class ComplexitySlotAdapter(nn.Module):
    """Route blocks through a complexity-scaled sparse slot hierarchy."""

    def __init__(
        self,
        hidden: int,
        *,
        max_slots: int = 4,
        min_slots: int = 2,
        assignment_top_k: int = 2,
        bottleneck: int | None = None,
    ) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        width = int(bottleneck or max(16, hidden // 4))
        self.hidden = int(hidden)
        self.bottleneck = width
        self.router = LearnedSparseSlotRouter(
            hidden,
            max_slots=max_slots,
            min_slots=min_slots,
            assignment_top_k=assignment_top_k,
        )
        self.state_encoder = nn.Linear(3, width)
        self.time_encoder = nn.Linear(1, width)
        self.reduce = nn.Sequential(
            nn.Linear(hidden * 2 + 2 * width, width),
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
        complexity: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, count, hidden = block_tokens.shape
        if hidden != self.hidden:
            raise ValueError("block token hidden size mismatch")
        if block_state.shape != (batch, count, 3):
            raise ValueError("block state must have shape (batch, n, 3)")
        if complexity.shape != (batch,):
            raise ValueError("complexity must have shape (batch,)")
        if time.shape != (batch,):
            raise ValueError("time must have shape (batch,)")
        slots, weights, active, assignments, router_diagnostics = self.router(
            block_tokens,
            complexity,
        )
        routed = torch.bmm(assignments, slots)
        time_code = self.time_encoder(time.float()[:, None])
        context = torch.cat(
            [
                block_tokens,
                routed,
                self.state_encoder(block_state.float()).to(block_tokens.dtype),
                time_code[:, None].expand(-1, count, -1).to(block_tokens.dtype),
            ],
            dim=-1,
        )
        residual = self.reduce(context)
        budget_scale = (
            router_diagnostics["budget_soft"] / float(self.router.max_slots)
        ).clamp(0.0, 1.0)
        residual = (
            self.gate
            * complexity[:, None, None]
            * budget_scale[:, None, None]
            * residual
        )
        diagnostics = {
            "complexity_slot_active_count": active.sum(dim=-1),
            "complexity_slot_weights": weights,
            "complexity_slot_assignment_entropy": router_diagnostics[
                "assignment_entropy"
            ],
            "complexity_slot_budget_soft": router_diagnostics["budget_soft"],
            "complexity_slot_residual_norm": residual.detach().norm(dim=-1),
        }
        return residual, diagnostics


__all__ = ["ComplexitySlotAdapter"]

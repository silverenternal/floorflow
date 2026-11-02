"""Fail-closed learned acceptance for candidate plan repairs."""

from __future__ import annotations

import torch
from torch import nn


class PlanRepairAcceptanceHead(nn.Module):
    """Score each candidate repair from plan and action evidence.

    The final bias starts at -6, so every candidate is rejected until the
    module is explicitly trained. This preserves the unmodified plan.
    """

    def __init__(
        self,
        plan_feature_dim: int,
        action_feature_dim: int,
        *,
        hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        self.plan = nn.Sequential(
            nn.Linear(int(plan_feature_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.action = nn.Sequential(
            nn.Linear(int(action_feature_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.score = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.constant_(self.score[-1].bias, -6.0)

    def forward(
        self,
        plan_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        if plan_features.ndim != 2 or action_features.ndim != 3:
            raise ValueError(
                "plan features must be [B,D] and actions [B,A,D]"
            )
        if plan_features.shape[0] != action_features.shape[0]:
            raise ValueError("plan and action batches must match")
        context = self.plan(plan_features.float())[:, None].expand(
            -1,
            action_features.shape[1],
            -1,
        )
        action = self.action(action_features.float())
        return self.score(torch.cat([context, action], dim=-1))[..., 0]


@torch.no_grad()
def select_accepted_repairs(
    accept_logits: torch.Tensor,
    *,
    threshold: float = 0.5,
    max_edits: int = 2,
) -> torch.Tensor:
    """Select at most a bounded number of accepted candidate repairs."""

    if accept_logits.ndim != 2:
        raise ValueError("accept logits must be [B,A]")
    probability = accept_logits.sigmoid()
    accepted = probability >= float(threshold)
    budget = max(0, int(max_edits))
    if budget < accepted.shape[1]:
        order = probability.masked_fill(~accepted, -1.0).argsort(
            dim=1,
            descending=True,
        )[:, :budget]
        selected = torch.zeros_like(accepted)
        selected.scatter_(1, order, True)
        accepted &= selected
    return accepted


__all__ = ["PlanRepairAcceptanceHead", "select_accepted_repairs"]

"""Presence-aware pooling for object-query representations."""

from __future__ import annotations

import torch
from torch import nn


class QueryAwarePool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(int(hidden_dim), 1)

    def forward(
        self,
        query_features: torch.Tensor,
        presence_logits: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_features.ndim != 3 or presence_logits.shape != query_features.shape[:2]:
            raise ValueError("expected query features [B,Q,D] and presence [B,Q]")
        if valid is None:
            valid = torch.ones_like(presence_logits, dtype=torch.bool)
        if valid.shape != presence_logits.shape or valid.dtype != torch.bool:
            raise ValueError("query valid mask must be boolean [B,Q]")
        logits = self.score(query_features).squeeze(-1) + presence_logits
        logits = logits.masked_fill(~valid, -torch.inf)
        has_valid = valid.any(dim=1, keepdim=True)
        safe_logits = logits.masked_fill(~has_valid, 0.0)
        weights = safe_logits.softmax(dim=1).masked_fill(~has_valid, 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return torch.einsum("bq,bqd->bd", weights, query_features)


__all__ = ["QueryAwarePool"]

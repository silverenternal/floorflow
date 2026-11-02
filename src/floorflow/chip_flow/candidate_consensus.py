"""Permutation-equivariant candidate fusion for portfolio distillation."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.candidate_legalizability import (
    CandidateLegalizabilityReranker,
)


class CandidateConsensusFuser(nn.Module):
    """Soft consensus or sparse selection with a bounded residual."""

    def __init__(
        self,
        hidden: int,
        *,
        candidate_feature_dim: int,
        context_dim: int = 0,
        top_k: int | None = None,
        selection_mode: str = "soft",
        residual_limit: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden <= 0 or candidate_feature_dim <= 0 or context_dim < 0:
            raise ValueError("invalid candidate fuser dimensions")
        if top_k is not None and top_k < 1:
            raise ValueError("candidate top_k must be positive")
        if selection_mode not in {"soft", "topk_residual"}:
            raise ValueError("selection_mode must be soft or topk_residual")
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")
        if selection_mode == "topk_residual" and top_k is None:
            top_k = 1
        self.hidden = int(hidden)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.context_dim = int(context_dim)
        self.top_k = int(top_k) if top_k is not None else None
        self.selection_mode = str(selection_mode)
        self.residual_limit = float(residual_limit)
        width = max(16, hidden // 8)
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(self.candidate_feature_dim),
            nn.Linear(self.candidate_feature_dim, width),
            nn.SiLU(),
        )
        self.state_encoder = nn.Linear(6, width)
        self.context_encoder = (
            nn.Linear(self.context_dim, width)
            if self.context_dim > 0
            else None
        )
        input_width = 3 * width if self.context_encoder is not None else 2 * width
        self.scorer = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        self.ranker = (
            CandidateLegalizabilityReranker(
                hidden=width,
                feature_dim=self.candidate_feature_dim,
            )
            if self.selection_mode == "topk_residual"
            else None
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def candidate_scores(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return lower-is-better legalizability scores for sparse selection."""

        if self.ranker is None:
            raise ValueError("candidate scores require topk_residual mode")
        if candidate_features.ndim != 3:
            raise ValueError("candidate features must have shape (batch, k, features)")
        if candidate_mask.shape != candidate_features.shape[:2]:
            raise ValueError("candidate mask shape mismatch")
        mask = candidate_mask.bool()
        safe_features = torch.where(
            mask[..., None],
            candidate_features.float(),
            candidate_features.new_zeros(()),
        )
        scores = self.ranker(safe_features)
        return scores.masked_fill(~mask, float("inf"))

    def _sparse_selection(
        self,
        candidate_states: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.candidate_scores(candidate_features, candidate_mask)
        logits = (-scores).masked_fill(~candidate_mask, float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        minimum_valid = int(candidate_mask.sum(dim=-1).min().item())
        selected_count = min(int(self.top_k or 1), minimum_valid)
        _, selected = probabilities.topk(selected_count, dim=-1)
        hard = torch.zeros_like(probabilities).scatter_(1, selected, 1.0)
        hard = torch.where(candidate_mask, hard, candidate_features.new_zeros(()))
        hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(1.0)
        straight_through = hard + probabilities - probabilities.detach()
        selected_state = torch.einsum("bk,bknd->bnd", hard, candidate_states)
        neighborhood = torch.einsum(
            "bk,bknd->bnd",
            probabilities,
            candidate_states,
        )
        delta = (neighborhood - selected_state) / self.residual_limit
        bounded = self.residual_limit * torch.tanh(delta)
        fused = selected_state + self.gate * bounded
        return fused, straight_through

    def forward(
        self,
        candidate_states: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_states.ndim != 4 or candidate_states.shape[-1] != 3:
            raise ValueError("candidate_states must have shape (batch, k, blocks, 3)")
        batch, count, _blocks, _ = candidate_states.shape
        if count == 0:
            raise ValueError("candidate consensus requires at least one candidate")
        if candidate_features.ndim != 3:
            raise ValueError(
                "candidate_features must have shape (batch, candidate, features)"
            )
        if candidate_features.shape[:2] != (batch, count):
            raise ValueError("candidate feature batch/candidate shape mismatch")
        if candidate_features.shape[-1] != self.candidate_feature_dim:
            raise ValueError("candidate feature width mismatch")
        if candidate_mask is None:
            candidate_mask = candidate_states.new_ones(batch, count, dtype=torch.bool)
        if candidate_mask.shape != (batch, count):
            raise ValueError("candidate mask shape mismatch")
        candidate_mask = candidate_mask.bool()
        if not bool(candidate_mask[:, 0].all()):
            raise ValueError("reference candidate must be valid in every batch row")
        if not bool(candidate_mask.any(dim=1).all()):
            raise ValueError("each batch row requires a valid candidate")
        if self.context_encoder is not None and (
            context is None or context.shape != (batch, self.context_dim)
        ):
            raise ValueError("candidate fuser context shape mismatch")
        zero_state = candidate_states.new_zeros(())
        safe_states = torch.where(
            candidate_mask[..., None, None],
            candidate_states.float(),
            zero_state,
        )
        safe_features = torch.where(
            candidate_mask[..., None],
            candidate_features.float(),
            zero_state,
        )
        if not torch.isfinite(safe_states).all() or not torch.isfinite(
            safe_features
        ).all():
            raise ValueError("valid candidate states and features must be finite")
        means = safe_states.mean(dim=2)
        variances = safe_states.var(dim=2, correction=0)
        state_features = torch.cat(
            [
                means,
                variances,
            ],
            dim=-1,
        )
        pieces = [
            self.feature_encoder(safe_features),
            self.state_encoder(state_features),
        ]
        if self.context_encoder is not None:
            context_code = self.context_encoder(context.float())
            pieces.append(context_code[:, None].expand(-1, count, -1))
        if self.selection_mode == "topk_residual":
            return self._sparse_selection(
                safe_states,
                safe_features,
                candidate_mask,
            )
        logits = self.scorer(torch.cat(pieces, dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, float("-inf"))
        if self.top_k is not None and self.top_k < count:
            threshold = logits.topk(self.top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        reference = candidate_states[:, 0]
        fused = torch.einsum("bk,bknd->bnd", weights, safe_states)
        return reference + self.gate * (fused - reference), weights


__all__ = ["CandidateConsensusFuser"]

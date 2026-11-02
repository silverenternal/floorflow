"""Hard-envelope constraint-aware candidate gating network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.model.constraint_state_encoder import ConstraintStateEncoder


class ConstraintAwareGate(nn.Module):
    """Rank feasible actions while preserving zero probability for invalid ones."""

    def __init__(
        self,
        node_features: int = 8,
        action_features: int = 15,
        hard_mask_features: int = 8,
        hidden_dim: int = 384,
        embedding_dim: int = 192,
        operations: int = 5,
        families: int = 5,
    ) -> None:
        super().__init__()
        self.state = ConstraintStateEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            operations=operations,
            families=families,
        )
        self.action_projection = nn.Sequential(
            nn.Linear(action_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hard_projection = nn.Sequential(
            nn.Linear(hard_mask_features, hidden_dim // 2),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        family_ids: torch.Tensor,
        operation_ids: torch.Tensor,
        action_features: torch.Tensor,
        hard_mask: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        global_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self.state(
            node_features,
            adjacency,
            family_ids,
            operation_ids,
            node_mask=node_mask,
            global_features=global_features,
            action_features=action_features,
            hard_mask=hard_mask,
        )
        action = self.action_projection(action_features.float())
        feasible = hard_mask.bool()
        hard = self.hard_projection(feasible.float())
        fused = self.fusion(
            torch.cat(
                [state["global_token"], action, hard],
                dim=-1,
            )
        )
        logits = self.score_head(fused).squeeze(-1)
        candidate_hard = hard_mask.bool().all(dim=-1)
        probability = candidate_hard.float() * logits.sigmoid()
        return {
            **state,
            "gate_logits": logits,
            "gate_probability": probability,
            "uncertainty": self.uncertainty_head(fused).squeeze(-1).sigmoid(),
            "hard_feasible": candidate_hard,
        }


def hard_masked_log_softmax(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Normalize logits only over valid candidates; invalid mass is exact zero."""

    if logits.shape != hard_mask.shape:
        raise ValueError("logits and hard_mask must have identical shapes")
    mask = hard_mask.bool()
    if not bool(mask.any(dim=dim).all()):
        raise ValueError("every candidate group must have one feasible action")
    masked = logits.masked_fill(~mask, -torch.inf)
    return F.log_softmax(masked, dim=dim)


__all__ = ["ConstraintAwareGate", "hard_masked_log_softmax"]

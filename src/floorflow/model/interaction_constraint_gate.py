"""Constraint-aware gate with observable multi-step interaction context."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.model.constraint_aware_gate import ConstraintAwareGate


class InteractionConstraintAwareGate(nn.Module):
    """Add a zero-initialized plan residual to the frozen-architecture CAGN."""

    def __init__(
        self,
        *,
        sequence_features: int = 12,
        sequence_hidden_dim: int = 256,
        interaction_scale: float = 10.0,
        **gate_config,
    ) -> None:
        super().__init__()
        self.interaction_scale = float(interaction_scale)
        self.base = ConstraintAwareGate(**gate_config)
        embedding_dim = int(gate_config.get("embedding_dim", 192))
        action_dim = int(gate_config.get("action_features", 15))
        hard_dim = int(gate_config.get("hard_mask_features", 8))
        node_dim = int(gate_config.get("node_features", 8))
        candidate_dim = (
            embedding_dim
            + action_dim
            + 15
            + hard_dim
            + 2 * node_dim
            + sequence_features
        )
        self.sequence_trunk = nn.Sequential(
            nn.Linear(candidate_dim, sequence_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(sequence_hidden_dim),
            nn.Linear(sequence_hidden_dim, sequence_hidden_dim),
            nn.SiLU(),
        )
        self.sequence_score_head = nn.Linear(sequence_hidden_dim, 1)
        self.sequence_value_head = nn.Linear(sequence_hidden_dim, 1)
        nn.init.zeros_(self.sequence_score_head.weight)
        nn.init.zeros_(self.sequence_score_head.bias)

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
        sequence_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.base(
            node_features,
            adjacency,
            family_ids,
            operation_ids,
            action_features,
            hard_mask,
            node_mask=node_mask,
            global_features=global_features,
        )
        if sequence_features is not None:
            if global_features is None:
                global_features = node_features.new_zeros(
                    node_features.shape[0],
                    15,
                )
            plan = torch.cat(
                [
                    output["embedding"],
                    action_features.float(),
                    global_features.float(),
                    hard_mask.float(),
                    node_features.float().mean(dim=1),
                    node_features.float().amax(dim=1),
                    sequence_features.float(),
                ],
                dim=-1,
            )
            encoded = self.sequence_trunk(plan)
            residual = self.sequence_score_head(encoded).squeeze(-1)
            residual = residual * (
                sequence_features[:, 1].float() > 0.0
            ).to(residual.dtype)
            residual = self.interaction_scale * residual
            output["interaction_value"] = self.sequence_value_head(encoded).squeeze(-1)
            logits = output["gate_logits"] + residual
            output["gate_logits"] = logits
            output["gate_probability"] = (
                output["hard_feasible"].float() * logits.sigmoid()
            )
        return output

    def load_base_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.base.load_state_dict(state_dict)


__all__ = ["InteractionConstraintAwareGate"]

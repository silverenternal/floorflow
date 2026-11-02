"""Dynamic hierarchical chip flow built from the existing slot primitives."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.model import BlockLayer
from floorflow.chip_flow.slot_model import ChipSlotFlowModel
from floorflow.representation.slots import soft_slot_decompose


class FeatureTokenEncoder(nn.Module):
    """Encode variable-length block features into contextual tokens."""

    def __init__(
        self,
        feature_dim: int,
        hidden: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.layers = nn.ModuleList(
            [BlockLayer(hidden, heads, None) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.input(features.float())
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm(tokens)


class HierarchicalSlotDecoder(nn.Module):
    """Decode block tokens from global state and dynamic condition slots."""

    def __init__(
        self,
        feature_dim: int,
        hidden: int,
        global_dim: int,
        layers: int,
        heads: int,
        num_slots: int,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.query = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.condition_tokens = FeatureTokenEncoder(
            feature_dim,
            hidden,
            layers=max(1, layers // 2),
            heads=heads,
        )
        self.cross_slot = nn.MultiheadAttention(
            hidden,
            heads,
            batch_first=True,
        )
        self.slot_gate = nn.Linear(hidden, 1)
        self.global_context = nn.Linear(global_dim * 2, hidden)
        self.layers = nn.ModuleList(
            [BlockLayer(hidden, heads, None) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(
        self,
        global_state: torch.Tensor,
        condition: torch.Tensor,
        features: torch.Tensor,
        bias: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del adjacency
        tokens = self.query(features.float())
        encoded_condition = self.condition_tokens(features)
        slots = soft_slot_decompose(
            encoded_condition,
            num_slots=self.num_slots,
        )
        attended, _weights = self.cross_slot(
            tokens,
            slots.slots,
            slots.slots,
            key_padding_mask=slots.active_slot_mask <= 0.0,
            need_weights=False,
        )
        gate = torch.sigmoid(self.slot_gate(attended))
        tokens = tokens + gate * attended
        context = self.global_context(
            torch.cat([global_state.float(), condition.float()], dim=-1)
        )
        tokens = tokens + context[:, None]
        for layer in self.layers:
            tokens = layer(tokens, bias)
        return self.head(self.norm(tokens))


class ChipHierarchicalSlotFlowModel(ChipSlotFlowModel):
    """Global CFM with dynamic hierarchical token-slot readout."""

    def __init__(
        self,
        *args,
        num_slots: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_slots = int(num_slots)
        self.target_hierarchy = FeatureTokenEncoder(
            self.feature_dim + 3,
            self.hidden,
            layers=max(1, int(kwargs.get("target_layers", 4)) // 2),
            heads=int(kwargs.get("heads", 6)),
        )
        self.target_slot_projection = nn.Linear(self.hidden, self.global_dim)
        nn.init.zeros_(self.target_slot_projection.weight)
        nn.init.zeros_(self.target_slot_projection.bias)
        self.decoder = HierarchicalSlotDecoder(
            feature_dim=self.feature_dim,
            hidden=self.hidden,
            global_dim=self.global_dim,
            layers=kwargs.get("decoder_layers", 4),
            heads=kwargs.get("heads", 6),
            num_slots=self.num_slots,
        )

    def encode_target(
        self,
        target_latent: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        global_state = super().encode_target(target_latent, features)
        tokens = self.target_hierarchy(
            torch.cat([target_latent.float(), features.float()], dim=-1)
        )
        slots = soft_slot_decompose(tokens, num_slots=self.num_slots)
        pooled = (slots.slots * slots.slot_weights.unsqueeze(-1)).sum(dim=1)
        return global_state + torch.tanh(self.target_slot_projection(pooled))

    @torch.no_grad()
    def active_slot_counts(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.decoder.condition_tokens(features.float())
        slots = soft_slot_decompose(tokens, num_slots=self.num_slots)
        return slots.active_slot_mask.sum(dim=1).long()


__all__ = [
    "ChipHierarchicalSlotFlowModel",
    "FeatureTokenEncoder",
    "HierarchicalSlotDecoder",
]

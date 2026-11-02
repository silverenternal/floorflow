"""Cross-channel and slot attention for semantic floorplan streams."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from floorflow.data.classes import semantic_channel_prior
from floorflow.representation.slots import gated_cross_attention, soft_slot_decompose


class SemanticChannelCrossAttention(nn.Module):
    """Exchange information across semantic drawing channels."""

    def __init__(
        self,
        hidden_channels: int,
        semantic_channels: int,
        heads: int = 4,
        use_relation_bias: bool = True,
        relation_prior: tuple[tuple[float, ...], ...] | None = None,
    ) -> None:
        super().__init__()
        if hidden_channels % heads:
            heads = 1
        self.semantic_queries = nn.Parameter(torch.randn(1, semantic_channels, hidden_channels) * 0.02)
        self.spatial_norm = nn.LayerNorm(hidden_channels)
        self.query_norm = nn.LayerNorm(hidden_channels)
        self.spatial_readout = nn.MultiheadAttention(hidden_channels, heads, batch_first=True)
        self.semantic_norm = nn.LayerNorm(hidden_channels)
        self.use_relation_bias = bool(use_relation_bias)
        if self.use_relation_bias:
            self.channel_query = nn.Linear(hidden_channels, hidden_channels)
            self.channel_key = nn.Linear(hidden_channels, hidden_channels)
            self.channel_value = nn.Linear(hidden_channels, hidden_channels)
            prior = torch.tensor(
                relation_prior if relation_prior is not None else semantic_channel_prior(),
                dtype=torch.float32,
            )
            if prior.shape != (semantic_channels, semantic_channels):
                prior = torch.zeros(semantic_channels, semantic_channels, dtype=torch.float32)
            self.channel_prior = nn.Parameter(prior)
            self.channel_gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )
        else:
            self.channel_attention = nn.MultiheadAttention(hidden_channels, heads, batch_first=True)
        self.semantic_feedback_norm = nn.LayerNorm(hidden_channels)
        self.spatial_feedback = nn.MultiheadAttention(hidden_channels, heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_channels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = hidden.shape
        spatial = hidden.flatten(2).transpose(1, 2)
        spatial_norm = self.spatial_norm(spatial)
        semantic = self.semantic_queries.to(hidden.dtype).expand(batch, -1, -1)
        semantic = semantic + self.spatial_readout(
            self.query_norm(semantic),
            spatial_norm,
            spatial_norm,
            need_weights=False,
        )[0]
        semantic_norm = self.semantic_norm(semantic)
        if self.use_relation_bias:
            query = self.channel_query(semantic_norm)
            key = self.channel_key(semantic_norm)
            value = self.channel_value(semantic_norm)
            scores = query @ key.transpose(1, 2) / math.sqrt(max(1, channels))
            scores = scores + self.channel_prior.to(device=scores.device, dtype=scores.dtype).unsqueeze(0)
            channel_attention = torch.softmax(scores, dim=-1)
            channel_context = channel_attention @ value
            channel_gate = torch.sigmoid(self.channel_gate(torch.cat([semantic, channel_context], dim=-1)))
            semantic = semantic + channel_gate * channel_context
        else:
            semantic = semantic + self.channel_attention(
                semantic_norm,
                semantic_norm,
                semantic_norm,
                need_weights=False,
            )[0]
        semantic_norm = self.semantic_feedback_norm(semantic)
        feedback = self.spatial_feedback(
            spatial_norm,
            semantic_norm,
            semantic_norm,
            need_weights=False,
        )[0]
        mixed = spatial + self.gate(torch.cat([spatial, feedback], dim=-1)) * feedback
        mixed = self.output_norm(mixed)
        return mixed.transpose(1, 2).reshape(batch, channels, height, width)


class SemanticSlotFlowMixer(nn.Module):
    """Multiflow-style slot decomposition and bidirectional gated attention."""

    def __init__(self, hidden_channels: int, num_slots: int = 16) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_slots = int(num_slots)
        self.spatial_norm = nn.LayerNorm(hidden_channels)
        self.condition_norm = nn.LayerNorm(hidden_channels)
        self.slot_refine = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.feedback_norm = nn.LayerNorm(hidden_channels)
        self.spatial_feedback = nn.MultiheadAttention(hidden_channels, 4, batch_first=True)
        self.feedback_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_channels)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, channels, height, width = hidden.shape
        spatial = self.spatial_norm(hidden.flatten(2).transpose(1, 2))
        condition_tokens = self.condition_norm(condition.flatten(2).transpose(1, 2))
        spatial_slots = soft_slot_decompose(spatial, num_slots=self.num_slots)
        condition_slots = soft_slot_decompose(condition_tokens, num_slots=self.num_slots)
        spatial_from_condition = gated_cross_attention(spatial_slots, condition_slots)
        condition_from_spatial = gated_cross_attention(condition_slots, spatial_slots)
        updated_spatial_slots = spatial_slots.slots + 0.5 * spatial_from_condition.attended
        updated_condition_slots = condition_slots.slots + 0.5 * condition_from_spatial.attended
        updated_spatial_slots = updated_spatial_slots + self.slot_refine(updated_condition_slots)
        feedback = self.spatial_feedback(
            spatial,
            self.feedback_norm(updated_spatial_slots),
            self.feedback_norm(updated_spatial_slots),
            need_weights=False,
        )[0]
        mixed = spatial + self.feedback_gate(torch.cat([spatial, feedback], dim=-1)) * feedback
        mixed = self.output_norm(mixed)
        output = mixed.transpose(1, 2).reshape(batch, channels, height, width)
        diagnostics = {
            "spatial_to_condition_gate": spatial_from_condition.gate,
            "condition_to_spatial_gate": condition_from_spatial.gate,
            "active_spatial_slots": spatial_slots.active_slot_mask.sum(dim=1).float().mean(),
        }
        return output, diagnostics

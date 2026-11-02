"""Cross-attention fusion over precomputed typed channel tokens."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.typed_channel_state import (
    TypedChannelState,
    TypedGraphOutput,
)
from floorflow.chip_flow.typed_cross_attention import TypedCrossAttention


def fuse_typed_channels(
    cross_attention: TypedCrossAttention,
    norm: nn.LayerNorm,
    state: TypedChannelState,
    *,
    relation_specific: bool,
) -> TypedGraphOutput:
    """Apply the independently trainable cross-attention stage."""

    memory = torch.cat(
        [
            state.global_token,
            state.structural_tokens,
            state.constraint_tokens,
            state.port_tokens,
        ],
        dim=1,
    )
    if relation_specific:
        if state.relation_memories is None or state.relation_masks is None:
            raise ValueError("relation-specific fusion requires relation masks")
        block_update = cross_attention.forward_relations(
            state.block_tokens,
            memory,
            state.allowed,
            state.relation_memories,
            state.relation_masks,
        )
    else:
        block_update = cross_attention(
            state.block_tokens,
            memory,
            state.allowed,
            group_sizes=state.group_sizes,
        )
    updated = norm(state.block_tokens + block_update)
    diagnostics = dict(state.diagnostics)
    if cross_attention.relation_gate is not None:
        magnitude = cross_attention.relation_gate.detach().square()
        diagnostics["relation_gate_magnitude"] = magnitude.mean()
        probability = torch.softmax(magnitude, dim=0)
        diagnostics["relation_gate_entropy"] = -(
            probability * probability.clamp_min(1.0e-8).log()
        ).sum()
    return TypedGraphOutput(
        block_tokens=updated,
        global_token=state.global_token,
        structural_tokens=state.structural_tokens,
        constraint_tokens=state.constraint_tokens,
        port_tokens=state.port_tokens,
        structural_mask=state.structural_mask,
        constraint_mask=state.constraint_mask,
        port_mask=state.port_mask,
        port_block_incidence=state.port_block_incidence,
        diagnostics=diagnostics,
    )


__all__ = ["fuse_typed_channels"]

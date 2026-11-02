"""Typed sparse cross-attention used by the chip graph encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TypedCrossAttention(nn.Module):
    """Zero-gated cross-attention over an explicitly allowed typed mask."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        *,
        relation_specific: bool = False,
        incidence_normalized_relations: bool = False,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.gate = nn.Parameter(torch.zeros(1))
        self.type_bias = nn.Parameter(torch.zeros(4))
        self.relation_gate = (
            nn.Parameter(torch.zeros(4)) if relation_specific else None
        )
        self.incidence_normalized_relations = bool(
            incidence_normalized_relations
        )

    def _attention(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, _ = query.shape
        key_count = memory.shape[1]
        q = self.q(query).reshape(batch, count, self.heads, self.head_dim)
        k = self.k(memory).reshape(batch, key_count, self.heads, self.head_dim)
        v = self.v(memory).reshape(batch, key_count, self.heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        mask = allowed[:, None]
        active = mask.any(dim=-1, keepdim=True)
        key_index = torch.arange(
            key_count,
            device=allowed.device,
        ).view(1, 1, 1, key_count)
        safe = mask | (~active & key_index.eq(0))
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=safe)
        mixed = mixed.transpose(1, 2).reshape(
            batch,
            count,
            self.heads * self.head_dim,
        )
        return mixed * active.squeeze(1).squeeze(-1).to(mixed.dtype)[..., None]

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        allowed: torch.Tensor,
        group_sizes: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if group_sizes is None:
            return self.gate * self.out(self._attention(query, memory, allowed))
        batch, count, hidden = query.shape
        key_count = memory.shape[1]
        q = self.q(query).reshape(batch, count, self.heads, self.head_dim)
        k = self.k(memory).reshape(batch, key_count, self.heads, self.head_dim)
        v = self.v(memory).reshape(batch, key_count, self.heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if len(group_sizes) != self.type_bias.numel():
            raise ValueError("expected one group size per type bias")
        if sum(group_sizes) != key_count:
            raise ValueError("typed group sizes must cover all memory tokens")
        bias = torch.zeros(
            batch,
            1,
            count,
            key_count,
            dtype=q.dtype,
            device=q.device,
        )
        bias = bias.masked_fill(~allowed[:, None], float("-inf"))
        start = 0
        for index, size in enumerate(group_sizes):
            bias[..., start : start + size] += self.type_bias[index]
            start += size
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        mixed = mixed.transpose(1, 2).reshape(batch, count, hidden)
        return self.gate * self.out(mixed)

    def _normalize_relation(
        self,
        mixed: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        if not self.incidence_normalized_relations:
            return mixed
        active = allowed.float().sum(dim=-1).squeeze(-1)
        return mixed * active.clamp_min(1.0).rsqrt()[..., None]

    def forward_relations(
        self,
        query: torch.Tensor,
        base_memory: torch.Tensor,
        base_allowed: torch.Tensor,
        memories: tuple[torch.Tensor, ...],
        masks: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Apply one shared projection independently per relation family."""

        del base_memory, base_allowed
        if self.relation_gate is None:
            raise ValueError("relation gates require relation-specific attention")
        if len(memories) != 4 or len(masks) != 4:
            raise ValueError("relation attention expects four memory families")
        mixed = torch.zeros_like(query)
        for index, (memory, allowed) in enumerate(zip(memories, masks, strict=True)):
            if memory.shape[1] == 0:
                continue
            mixed = mixed + self.relation_gate[index] * self._normalize_relation(
                self._attention(query, memory, allowed),
                allowed,
            )
        return self.gate * self.out(mixed)


__all__ = ["TypedCrossAttention"]

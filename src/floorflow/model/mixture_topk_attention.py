"""PyTorch MiTA-style cross attention without a FlashAttention dependency."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureTopKAttention(nn.Module):
    """Landmark-routed top-k attention for floorplan token streams.

    This follows MiTA's core mechanism: landmark queries build reusable
    top-k key/value sets, query tokens select a landmark, and a shared
    landmark value acts as a global expert.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        router_count: int = 16,
        top_k: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if router_count <= 0 or top_k <= 0:
            raise ValueError("router_count and top_k must be positive")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.router_count = int(router_count)
        self.top_k = int(top_k)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query_tokens: torch.Tensor,
        key_value_tokens: torch.Tensor,
        router_prior: torch.Tensor | None = None,
        router_budgets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if query_tokens.ndim != 3 or key_value_tokens.ndim != 3:
            raise ValueError("tokens must be [B,N,D]")
        if query_tokens.shape[-1] != self.dim or key_value_tokens.shape[-1] != self.dim:
            raise ValueError("token dimensions do not match attention width")
        batch, query_count, _ = query_tokens.shape
        key_count = key_value_tokens.shape[1]
        router_count = min(self.router_count, query_count)
        max_top_k = min(self.top_k, key_count)
        q = self._heads(self.q_proj(query_tokens))
        k = self._heads(self.k_proj(key_value_tokens))
        v = self._heads(self.v_proj(key_value_tokens))
        router = F.adaptive_avg_pool1d(
            q.permute(0, 1, 3, 2).reshape(
                batch * self.num_heads,
                self.head_dim,
                query_count,
            ),
            router_count,
        ).reshape(batch, self.num_heads, self.head_dim, router_count)
        router = router.permute(0, 1, 3, 2)
        agent_logits = torch.einsum("bhmd,bhnd->bhmn", router, k) / math.sqrt(
            self.head_dim
        )
        agent_weights = torch.softmax(agent_logits, dim=-1)
        agent_value = torch.einsum("bhmn,bhnd->bhmd", agent_weights, v)
        router_key_scores = torch.einsum("bhmd,bhnd->bhmn", router, k)
        query_router_scores = torch.einsum("bhmd,bhnd->bhmn", router, q)
        if router_prior is not None:
            if router_prior.shape != (batch, query_count, router_count):
                raise ValueError("router_prior must be [B,N,R]")
            query_router_scores = query_router_scores + router_prior.transpose(
                1,
                2,
            )[:, None].to(query_router_scores.dtype)
        if router_budgets is None:
            budgets = torch.full(
                (batch, router_count),
                max_top_k,
                device=query_tokens.device,
                dtype=torch.long,
            )
        else:
            if router_budgets.shape != (batch, router_count):
                raise ValueError("router_budgets must be [B,R]")
            budgets = router_budgets.to(
                device=query_tokens.device,
                dtype=torch.long,
            ).clamp(1, max_top_k)
        top_indices = router_key_scores.topk(max_top_k, dim=-1).indices
        rank = torch.arange(max_top_k, device=query_tokens.device)
        top_valid = rank[None, None, None, :] < budgets[:, None, :, None]
        router_index = query_router_scores.argmax(dim=2)
        gather_index = (
            router_index.unsqueeze(-1) * max_top_k
            + rank[None, None, None, :]
        )
        selected_indices = torch.gather(
            top_indices.reshape(
                batch,
                self.num_heads,
                router_count * max_top_k,
            ),
            2,
            gather_index.reshape(batch, self.num_heads, -1),
        ).reshape(batch, self.num_heads, query_count, max_top_k)
        selected_valid = torch.gather(
            top_valid.expand(
                batch,
                self.num_heads,
                router_count,
                max_top_k,
            ).reshape(
                batch,
                self.num_heads,
                router_count * max_top_k,
            ),
            2,
            gather_index.reshape(batch, self.num_heads, -1),
        ).reshape(batch, self.num_heads, query_count, max_top_k)
        safe_indices = selected_indices.clamp(0, key_count - 1)
        expand = safe_indices.reshape(
            batch,
            self.num_heads,
            query_count * max_top_k,
        ).unsqueeze(-1).expand(
            batch,
            self.num_heads,
            query_count * max_top_k,
            self.head_dim,
        )
        selected_key = torch.gather(k, 2, expand).reshape(
            batch, self.num_heads, query_count, max_top_k, self.head_dim
        )
        selected_value = torch.gather(v, 2, expand).reshape(
            batch, self.num_heads, query_count, max_top_k, self.head_dim
        )
        agent_key = torch.gather(
            router,
            2,
            router_index.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
        )
        agent_selected_value = torch.gather(
            agent_value,
            2,
            router_index.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
        )
        keys = torch.cat([selected_key, agent_key.unsqueeze(3)], dim=3)
        values = torch.cat([selected_value, agent_selected_value.unsqueeze(3)], dim=3)
        logits = torch.einsum("bhnd,bhnkd->bhnk", q, keys) / math.sqrt(
            self.head_dim
        )
        valid_keys = torch.cat(
            [
                selected_valid,
                torch.ones(
                    batch,
                    self.num_heads,
                    query_count,
                    1,
                    device=query_tokens.device,
                    dtype=torch.bool,
                ),
            ],
            dim=-1,
        )
        logits = logits.masked_fill(~valid_keys[:, :, :1, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        output = torch.einsum("bhnk,bhnkd->bhnd", weights, values)
        output = output.transpose(1, 2).reshape(batch, query_count, self.dim)
        output = self.dropout(self.out_proj(output))
        routing = torch.softmax(query_router_scores, dim=2).transpose(-1, -2)
        usage = F.one_hot(router_index, router_count).float().mean(dim=(0, 2))
        routing_entropy = _normalized_entropy(routing, dim=-1)
        attention_entropy = _normalized_entropy(weights, dim=-1)
        router_usage = usage.mean(dim=0)
        effective_experts = torch.exp(
            -(
                router_usage.clamp_min(1.0e-8)
                * router_usage.clamp_min(1.0e-8).log()
            ).sum()
        )
        return output, {
            "routing": routing,
            "router_index": router_index,
            "router_usage": router_usage,
            "router_usage_per_head": usage,
            "routing_entropy": routing_entropy.mean(),
            "attention_entropy": attention_entropy.mean(),
            "effective_expert_fraction": effective_experts / router_count,
            "top_k": budgets.float().mean(),
            "router_budget_min": budgets.float().amin(),
            "router_budget_max": budgets.float().amax(),
        }


def _normalized_entropy(probability: torch.Tensor, dim: int) -> torch.Tensor:
    cardinality = probability.shape[dim]
    if cardinality <= 1:
        return torch.ones((), device=probability.device, dtype=probability.dtype)
    entropy = -(
        probability.clamp_min(1.0e-8) * probability.clamp_min(1.0e-8).log()
    ).sum(dim=dim)
    return entropy / math.log(cardinality)


__all__ = ["MixtureTopKAttention"]

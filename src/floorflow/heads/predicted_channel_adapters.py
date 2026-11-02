"""Dynamic query and source attention adapters for predicted channels."""

from __future__ import annotations

import math

import torch
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.typed_channel_graph import (
    FURNITURE,
    OPENING,
    ROOM,
    WALL,
    channel_family_ids,
)


def typed_source_prior(
    taxonomy: FloorplanTaxonomy,
    source_count: int,
) -> torch.Tensor:
    """Build the structural channel-to-source attention prior."""

    families = channel_family_ids(taxonomy)
    prior = torch.zeros(taxonomy.num_classes, int(source_count))
    for family, values in (
        (ROOM, (1.2, 1.0, 0.0, 0.5)),
        (WALL, (1.0, 1.5, 0.0, 0.5)),
        (OPENING, (0.9, 1.5, 0.0, 0.5)),
        (FURNITURE, (1.0, 0.5, 1.5, 0.3)),
    ):
        prior[families == family, :4] = torch.tensor(values)
    if int(source_count) >= 6:
        prior[families == ROOM, 4:] = torch.tensor([1.3, 0.8])
        prior[families == WALL, 4:] = torch.tensor([0.5, 1.0])
        prior[families == OPENING, 4:] = torch.tensor([0.4, 1.0])
        prior[families == FURNITURE, 4:] = torch.tensor([0.8, 0.3])
    return prior


class DynamicChannelQueryAdapter(nn.Module):
    """Generate a sample-conditioned offset for every semantic channel."""

    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        source_dim: int,
        source_count: int,
        *,
        hyper_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.channel_dim = int(channel_dim)
        self.source_count = int(source_count)
        self.source_norm = nn.LayerNorm(int(source_dim))
        self.hyper = nn.Sequential(
            nn.Linear(self.source_count * int(source_dim), int(hyper_hidden)),
            nn.SiLU(),
            nn.Linear(
                int(hyper_hidden),
                self.num_channels * self.channel_dim,
            ),
        )
        self.scale = nn.Parameter(torch.tensor(0.01))
        nn.init.normal_(self.hyper[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.hyper[-1].bias)

    def forward(
        self,
        queries: torch.Tensor,
        source_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if source_tokens.shape[1] != self.source_count:
            raise ValueError("source token count does not match adapter")
        source = self.source_norm(source_tokens).flatten(1)
        offset = self.hyper(source).reshape(
            source.shape[0],
            self.num_channels,
            self.channel_dim,
        )
        return queries + torch.tanh(self.scale) * offset


class SourceConditionedChannelAttention(nn.Module):
    """Route each channel query over typed predicted-control source evidence."""

    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        source_dim: int,
        *,
        source_prior: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.channel_dim = int(channel_dim)
        self.source_projection = nn.Linear(int(source_dim), self.channel_dim)
        self.source_norm = nn.LayerNorm(self.channel_dim)
        self.query = nn.Linear(self.channel_dim, self.channel_dim)
        self.key = nn.Linear(self.channel_dim, self.channel_dim)
        self.value = nn.Linear(self.channel_dim, self.channel_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.channel_dim * 2, self.channel_dim),
            nn.SiLU(),
            nn.Linear(self.channel_dim, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))
        prior = (
            torch.zeros(self.num_channels, source_dim)
            if source_prior is None
            else source_prior.float()
        )
        if prior.ndim != 2 or prior.shape[0] != self.num_channels:
            raise ValueError("source prior shape does not match channel/source")
        self.register_buffer("source_prior", prior)
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        queries: torch.Tensor,
        source_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory = self.source_norm(self.source_projection(source_tokens))
        logits = torch.einsum(
            "bqd,bsd->bqs",
            self.query(queries),
            self.key(memory),
        ) / math.sqrt(float(queries.shape[-1]))
        logits = logits + self.source_prior[None].to(logits.dtype)
        weights = logits.softmax(dim=-1)
        context = torch.einsum("bqs,bsd->bqd", weights, self.value(memory))
        gate = torch.sigmoid(self.gate(torch.cat([queries, context], dim=-1)))
        output = queries + torch.tanh(self.scale) * gate * context
        entropy = -(
            weights.clamp_min(1.0e-8) * weights.clamp_min(1.0e-8).log()
        ).sum(dim=-1)
        entropy = entropy / max(1.0, math.log(max(1, weights.shape[-1])))
        return output, {
            "source_attention": weights,
            "source_attention_gate": gate,
            "source_attention_entropy": entropy.mean(),
        }


class ChannelGroupQueryAdapter(nn.Module):
    """Train one semantic channel family without changing shared queries."""

    def __init__(
        self,
        query_count: int,
        channel_dim: int,
        source_dim: int,
        source_count: int,
    ) -> None:
        super().__init__()
        self.delta = nn.Parameter(
            torch.randn(query_count, channel_dim) * 1.0e-3
        )
        self.feature_scale = nn.Parameter(torch.ones(query_count, channel_dim))
        self.feature_shift = nn.Parameter(torch.zeros(query_count, channel_dim))
        self.source_gate = nn.Sequential(
            nn.LayerNorm(int(source_dim) * int(source_count)),
            nn.Linear(int(source_dim) * int(source_count), query_count),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.source_gate[1].bias)

    def forward(
        self,
        query: torch.Tensor,
        source_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.source_gate(source_tokens.flatten(1))
        adapted = (
            query * self.feature_scale
            + self.feature_shift
            + torch.tanh(self.scale) * gate[..., None] * self.delta[None]
        )
        return adapted, gate


class ChannelGroupQueryAdapters(nn.Module):
    """Own independently trainable query residuals for CAD channel families."""

    def __init__(
        self,
        taxonomy: FloorplanTaxonomy,
        channel_dim: int,
        source_dim: int,
        source_count: int,
        enabled: bool,
    ) -> None:
        super().__init__()
        families = channel_family_ids(taxonomy)
        self.indices = {
            name: tuple(int(value) for value in (families == family).nonzero().flatten())
            for name, family in (
                ("room", ROOM),
                ("wall", WALL),
                ("opening", OPENING),
                ("furniture", FURNITURE),
            )
        }
        self.adapters = nn.ModuleDict(
            {
                name: ChannelGroupQueryAdapter(
                    len(indices),
                    channel_dim,
                    source_dim,
                    source_count,
                )
                for name, indices in self.indices.items()
                if enabled and indices
            }
        )

    def forward(
        self,
        queries: torch.Tensor,
        source_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = queries.clone()
        diagnostics = {}
        for name, adapter in self.adapters.items():
            indices = torch.tensor(self.indices[name], device=queries.device)
            adapted, gate = adapter(
                output[:, indices],
                source_tokens.to(queries.dtype),
            )
            output[:, indices] = adapted
            diagnostics[name] = gate
        return output, diagnostics


def apply_channel_group_queries(
    head: nn.Module,
    queries: torch.Tensor,
    source_tokens: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """Apply optional channel-family adapters without head-type branching."""

    adapters = getattr(head, "channel_group_adapters", None)
    if adapters is None or not adapters.adapters:
        return queries, None
    if source_tokens is None:
        raise ValueError("source tokens are required for group adapters")
    return adapters(queries, source_tokens.to(queries.dtype))


class CrossAttentionLoRA(nn.Module):
    """Low-rank Q/KV input residuals and output residual for cross-attention."""

    def __init__(self, dim: int, rank: int = 8) -> None:
        super().__init__()
        rank = max(1, int(rank))
        self.query_down = nn.Linear(dim, rank, bias=False)
        self.query_up = nn.Linear(rank, dim, bias=False)
        self.kv_down = nn.Linear(dim, rank, bias=False)
        self.kv_up = nn.Linear(rank, dim, bias=False)
        self.output_down = nn.Linear(dim, rank, bias=False)
        self.output_up = nn.Linear(rank, dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.rank_logits = nn.Parameter(torch.zeros(3, rank))
        for module in (
            self.query_up,
            self.kv_up,
            self.output_up,
        ):
            nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)

    def adapt_inputs(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.tanh(self.scale)
        gates = self.rank_gates()
        return (
            query + scale * self.query_up(
                self.query_down(query) * gates[0]
            ),
            memory + scale * self.kv_up(
                self.kv_down(memory) * gates[1]
            ),
        )

    def adapt_output(self, context: torch.Tensor) -> torch.Tensor:
        return context + torch.tanh(self.scale) * self.output_up(
            self.output_down(context) * self.rank_gates()[2]
        )

    def rank_gates(self) -> torch.Tensor:
        target = max(1, self.rank_logits.shape[1] // 2)
        if self.training:
            weights = torch.sigmoid(self.rank_logits)
            return weights * (
                float(target) / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            )
        indices = self.rank_logits.topk(target, dim=1).indices
        gates = torch.zeros_like(self.rank_logits)
        gates.scatter_(1, indices, 1.0)
        return gates

    def rank_budget_loss(self) -> torch.Tensor:
        target = max(1, self.rank_logits.shape[1] // 2)
        return (self.rank_gates().sum(dim=1) - float(target)).square().mean()


class SpatialConvAdapter(nn.Module):
    """Zero-initialized bottleneck depthwise adapter for channel features."""

    def __init__(
        self,
        channels: int,
        *,
        bottleneck: int = 16,
    ) -> None:
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.depthwise = nn.Conv2d(
            bottleneck,
            bottleneck,
            3,
            padding=1,
            groups=bottleneck,
        )
        self.up = nn.Conv2d(bottleneck, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.up.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.up(
            torch.nn.functional.silu(self.depthwise(self.down(feature)))
        )
        return feature + torch.tanh(self.scale) * residual


__all__ = [
    "ChannelGroupQueryAdapter",
    "ChannelGroupQueryAdapters",
    "CrossAttentionLoRA",
    "DynamicChannelQueryAdapter",
    "SourceConditionedChannelAttention",
    "SpatialConvAdapter",
    "apply_channel_group_queries",
    "typed_source_prior",
]

"""Masked channel-query distribution head adapted from EoMT (CVPR 2025)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.deterministic_pooling import fixed_grid_pool
from floorflow.heads.attention_masks import soft_pixel_attention_bias
from floorflow.heads.predicted_channel_adapters import (
    ChannelGroupQueryAdapters,
    CrossAttentionLoRA,
    DynamicChannelQueryAdapter,
    SourceConditionedChannelAttention,
    SpatialConvAdapter,
    apply_channel_group_queries,
    typed_source_prior,
)
from floorflow.model.typed_channel_graph import typed_channel_prior


def _position_embedding(height: int, width: int, channels: int, device):
    if channels % 4:
        raise ValueError("position embedding channels must be divisible by 4")
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(channels // 4, device=device)
        / max(1, channels // 4 - 1)
    )
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device),
        torch.linspace(0.0, 1.0, width, device=device),
        indexing="ij",
    )
    x_encoding = torch.cat(
        [torch.sin(xx[..., None] * frequency), torch.cos(xx[..., None] * frequency)],
        dim=-1,
    )
    y_encoding = torch.cat(
        [torch.sin(yy[..., None] * frequency), torch.cos(yy[..., None] * frequency)],
        dim=-1,
    )
    return torch.cat([x_encoding, y_encoding], dim=-1).flatten(0, 1)[None]


class MaskedChannelLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        mlp_ratio: float = 2.0,
        *,
        dynamic_cross_gate: bool = False,
        cross_attention_lora_rank: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = int(heads)
        self.self_attention = nn.MultiheadAttention(
            hidden,
            heads,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden,
            heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, int(hidden * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden * mlp_ratio), hidden),
        )
        self.cross_gate = (
            nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            if dynamic_cross_gate
            else None
        )
        if self.cross_gate is not None:
            nn.init.normal_(
                self.cross_gate[-1].weight,
                mean=0.0,
                std=1.0e-3,
            )
            nn.init.zeros_(self.cross_gate[-1].bias)
        self.cross_attention_lora = (
            CrossAttentionLoRA(hidden, int(cross_attention_lora_rank))
            if int(cross_attention_lora_rank) > 0
            else None
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        self_attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if attention_mask is not None:
            attention_mask = attention_mask[:, None].expand(
                -1,
                self.num_heads,
                -1,
                -1,
            ).reshape(
                attention_mask.shape[0] * self.num_heads,
                attention_mask.shape[1],
                attention_mask.shape[2],
            )
        normalized = self.norm1(queries)
        self_mask = None
        if self_attention_bias is not None:
            self_mask = self_attention_bias[:, None].expand(
                -1,
                self.num_heads,
                -1,
                -1,
            ).reshape(
                self_attention_bias.shape[0] * self.num_heads,
                self_attention_bias.shape[1],
                self_attention_bias.shape[2],
            )
        queries = queries + self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=self_mask,
            need_weights=False,
        )[0]
        cross_query = self.norm2(queries)
        cross_memory = memory
        if self.cross_attention_lora is not None:
            cross_query, cross_memory = self.cross_attention_lora.adapt_inputs(
                cross_query,
                cross_memory,
            )
        attended = self.cross_attention(
            cross_query,
            cross_memory,
            cross_memory,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        if self.cross_attention_lora is not None:
            attended = self.cross_attention_lora.adapt_output(attended)
        cross_gate = None
        if self.cross_gate is not None:
            cross_gate = torch.sigmoid(
                self.cross_gate(torch.cat([queries, attended], dim=-1))
            )
            attended = cross_gate * attended
        queries = queries + attended
        return queries + self.ffn(self.norm3(queries)), cross_gate


class MaskedChannelDistributionHead(nn.Module):
    """Predict channel masks with masked channel-pixel cross-attention."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 96,
        heads: int = 4,
        layers: int = 2,
        grid: int = 16,
        full_resolution_output: bool = False,
        context_dim: int = 0,
        dynamic_channel_queries: bool = False,
        source_attention: bool = False,
        dynamic_cross_attention: bool = False,
        cross_attention_lora_rank: int = 0,
        source_count: int = 4,
        source_hyper_hidden: int = 64,
        spatial_conv_adapter: bool = False,
        spatial_adapter_bottleneck: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.foreground_channels = self.num_classes - 1
        self.source_count = int(source_count)
        self.grid = int(grid)
        self.full_resolution_output = bool(full_resolution_output)
        if hidden_channels % heads:
            raise ValueError("hidden channels must be divisible by heads")
        self.pixel_projection = nn.Sequential(
            nn.Conv2d(int(in_channels), hidden_channels, 1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )
        self.spatial_adapter = (
            SpatialConvAdapter(
                hidden_channels,
                bottleneck=spatial_adapter_bottleneck,
            )
            if spatial_conv_adapter
            else None
        )
        self.channel_queries = nn.Parameter(
            torch.randn(self.num_classes, hidden_channels) * 0.02
        )
        self.context_projection = (
            nn.Sequential(
                nn.Linear(int(context_dim), hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            if context_dim > 0
            else None
        )
        if self.context_projection is not None:
            nn.init.normal_(self.context_projection[-1].weight, std=0.01)
            nn.init.zeros_(self.context_projection[-1].bias)
        self.layers = nn.ModuleList(
            [
                MaskedChannelLayer(
                    hidden_channels,
                    heads,
                    dynamic_cross_gate=dynamic_cross_attention,
                    cross_attention_lora_rank=cross_attention_lora_rank,
                )
                for _ in range(layers)
            ]
        )
        self.dynamic_channel_queries = bool(dynamic_channel_queries)
        self.dynamic_query_adapter = (
            DynamicChannelQueryAdapter(
                self.num_classes,
                hidden_channels,
                int(in_channels),
                self.source_count,
                hyper_hidden=source_hyper_hidden,
            )
            if self.dynamic_channel_queries
            else None
        )
        self.source_attention = bool(source_attention)
        self.source_adapter = (
            SourceConditionedChannelAttention(
                self.num_classes,
                hidden_channels,
                int(in_channels),
                source_prior=torch.zeros(
                    self.num_classes,
                    self.source_count,
                ),
            )
            if self.source_attention
            else None
        )
        self.norm = nn.LayerNorm(hidden_channels)
        self.mask_embedding = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.output_projection = (
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
                nn.SiLU(),
            )
            if self.full_resolution_output
            else nn.Identity()
        )
        self.mask_logit_scale = nn.Parameter(torch.tensor(5.0))
        self.background_logit = nn.Parameter(torch.full((1,), -2.0))
        self.reliability_head: nn.Module | None = None

    def _masked_memory(
        self,
        mask_logits: torch.Tensor | None,
        size: tuple[int, int],
    ) -> torch.Tensor | None:
        return soft_pixel_attention_bias(mask_logits, size)

    def channel_attention_bias(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        queries: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        del queries, memory
        return None

    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor | None = None,
        source_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Keep the promoted field projection in parameter dtype on Blackwell.
        with torch.autocast(device_type=features.device.type, enabled=False):
            projection = self.pixel_projection(features.float())
        if self.spatial_adapter is not None:
            projection = self.spatial_adapter(projection)
        space = fixed_grid_pool(projection, (self.grid, self.grid))
        memory = space.flatten(2).transpose(1, 2)
        memory = memory + _position_embedding(
            self.grid,
            self.grid,
            memory.shape[-1],
            memory.device,
        ).to(memory.dtype)
        queries = self.channel_queries[None].expand(features.shape[0], -1, -1)
        if self.context_projection is not None and context is not None:
            queries = queries + self.context_projection(
                context.to(queries.dtype)
            )[:, None]
        source_diagnostics = None
        if self.dynamic_query_adapter is not None:
            if source_tokens is None:
                raise ValueError("source tokens are required for dynamic queries")
            queries = self.dynamic_query_adapter(
                queries,
                source_tokens.to(queries.dtype),
            )
        if self.source_adapter is not None:
            if source_tokens is None:
                raise ValueError("source tokens are required for source attention")
            queries, source_diagnostics = self.source_adapter(
                queries,
                source_tokens.to(queries.dtype),
            )
        queries, group_adapter_diagnostics = apply_channel_group_queries(
            self,
            queries,
            source_tokens,
        )
        channel_bias = None
        mask_logits = None
        cross_gates = []
        for layer in self.layers:
            channel_bias = self.channel_attention_bias(
                features.shape[0],
                features.device,
                features.dtype,
                queries=queries,
                memory=memory,
            )
            attention_mask = self._masked_memory(
                mask_logits,
                (self.grid, self.grid),
            )
            queries, cross_gate = layer(
                queries,
                memory,
                attention_mask,
                self_attention_bias=channel_bias,
            )
            if cross_gate is not None:
                cross_gates.append(cross_gate.mean())
            mask_logits = torch.einsum(
                "bqc,bnc->bqn",
                self.mask_embedding(self.norm(queries)),
                memory,
            ).reshape(
                features.shape[0],
                self.num_classes,
                self.grid,
                self.grid,
            )
        if self.full_resolution_output:
            mask_tokens = F.normalize(
                self.mask_embedding(self.norm(queries)),
                dim=-1,
            )
            pixel_tokens = F.normalize(
                self.output_projection(projection),
                dim=1,
            )
            mask_logits = torch.einsum(
                "bqc,bchw->bqhw",
                mask_tokens,
                pixel_tokens,
            ) * self.mask_logit_scale
        else:
            mask_logits = F.interpolate(
                mask_logits,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        foreground_logits = mask_logits[:, 1:]
        channel_reliability = None
        if self.reliability_head is not None:
            channel_reliability = torch.sigmoid(
                self.reliability_head(self.norm(queries))
            ).squeeze(-1)
            foreground_logits = foreground_logits + channel_reliability.clamp_min(
                1.0e-5
            )[:, 1:].log()[:, :, None, None]
        background_logits = (
            mask_logits[:, :1]
            * 0.05
            + self.background_logit.to(mask_logits.dtype)[None, :, None, None]
        )
        full_logits = torch.cat([background_logits, foreground_logits], dim=1)
        probability = full_logits.softmax(dim=1)
        entropy = -(
            probability.clamp_min(1.0e-8)
            * probability.clamp_min(1.0e-8).log()
        ).sum(dim=1, keepdim=True)
        reliability = 1.0 - entropy / max(1.0e-6, math.log(self.num_classes))
        return {
            "logits": full_logits,
            "probability": probability,
            "room_logits": full_logits[:, 1:],
            "mask_logits": mask_logits,
            "reliability": reliability,
            "channel_reliability": channel_reliability,
            "channel_tokens": self.norm(queries),
            "channel_attention_bias": (
                None if channel_bias is None else channel_bias.detach()
            ),
            "channel_cross_attention_gate": (
                torch.stack(cross_gates).mean()
                if cross_gates
                else None
            ),
            "channel_source_attention": (
                source_diagnostics["source_attention"].detach()
                if source_diagnostics is not None
                else None
            ),
            "channel_source_attention_gate": (
                source_diagnostics["source_attention_gate"].detach()
                if source_diagnostics is not None
                else None
            ),
            "channel_source_attention_entropy": (
                source_diagnostics["source_attention_entropy"].detach()
                if source_diagnostics is not None
                else None
            ),
            "channel_group_adapter_gates": group_adapter_diagnostics,
        }


class TypedChannelDistributionHead(MaskedChannelDistributionHead):
    """All-channel queries with learned typed non-uniform channel links."""

    def __init__(
        self,
        in_channels: int,
        taxonomy: FloorplanTaxonomy,
        hidden_channels: int = 96,
        heads: int = 4,
        layers: int = 2,
        grid: int = 16,
        full_resolution_output: bool = False,
        context_dim: int = 0,
        dynamic_channel_links: bool = False,
        dynamic_channel_queries: bool = False,
        source_attention: bool = False,
        dynamic_cross_attention: bool = False,
        channel_group_adapters: bool = False,
        cross_attention_lora_rank: int = 0,
        spatial_conv_adapter: bool = False,
        spatial_adapter_bottleneck: int = 16,
        source_count: int = 4,
        source_hyper_hidden: int = 64,
    ) -> None:
        super().__init__(
            in_channels,
            taxonomy.num_classes,
            hidden_channels=hidden_channels,
            heads=heads,
            layers=layers,
            grid=grid,
            full_resolution_output=full_resolution_output,
            context_dim=context_dim,
            dynamic_channel_queries=dynamic_channel_queries,
            source_attention=source_attention,
            dynamic_cross_attention=dynamic_cross_attention,
            cross_attention_lora_rank=cross_attention_lora_rank,
            spatial_conv_adapter=spatial_conv_adapter,
            spatial_adapter_bottleneck=spatial_adapter_bottleneck,
            source_count=source_count,
            source_hyper_hidden=source_hyper_hidden,
        )
        prior = typed_channel_prior(taxonomy)
        if prior.shape != (self.num_classes, self.num_classes):
            raise ValueError("typed channel prior size does not match taxonomy")
        self.register_buffer("typed_channel_prior", prior)
        source_prior = typed_source_prior(taxonomy, self.source_count)
        if self.source_adapter is not None:
            self.source_adapter.source_prior.copy_(source_prior)
        self.channel_group_adapters = ChannelGroupQueryAdapters(
            taxonomy,
            hidden_channels,
            int(in_channels),
            self.source_count,
            enabled=channel_group_adapters,
        )
        self.typed_channel_residual = nn.Parameter(torch.zeros_like(prior))
        self.typed_channel_scale = nn.Parameter(torch.tensor(0.5))
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        nn.init.normal_(
            self.reliability_head[-1].weight,
            mean=0.0,
            std=1.0e-3,
        )
        nn.init.zeros_(self.reliability_head[-1].bias)
        self.dynamic_channel_links = bool(dynamic_channel_links)
        self.dynamic_channel_norm = (
            nn.LayerNorm(hidden_channels)
            if self.dynamic_channel_links
            else None
        )
        self.dynamic_channel_query = (
            nn.Linear(hidden_channels, hidden_channels, bias=False)
            if self.dynamic_channel_links
            else None
        )
        self.dynamic_channel_key = (
            nn.Linear(hidden_channels, hidden_channels, bias=False)
            if self.dynamic_channel_links
            else None
        )
        self.dynamic_channel_memory = (
            nn.Linear(hidden_channels, hidden_channels, bias=False)
            if self.dynamic_channel_links
            else None
        )
        self.dynamic_channel_scale = (
            nn.Parameter(torch.tensor(0.01))
            if self.dynamic_channel_links
            else None
        )

    def channel_attention_bias(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        queries: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bias = self.typed_channel_prior + self.typed_channel_residual
        bias = bias * torch.tanh(self.typed_channel_scale)
        output = bias[None].expand(batch, -1, -1)
        if (
            self.dynamic_channel_links
            and queries is not None
            and self.dynamic_channel_query is not None
            and self.dynamic_channel_key is not None
            and self.dynamic_channel_memory is not None
            and self.dynamic_channel_norm is not None
            and self.dynamic_channel_scale is not None
            and memory is not None
        ):
            memory_context = self.dynamic_channel_memory(
                memory.mean(dim=1)
            )[:, None]
            normalized = self.dynamic_channel_norm(
                queries + memory_context
            ).float()
            query = self.dynamic_channel_query(normalized)
            key = self.dynamic_channel_key(normalized)
            dynamic = torch.bmm(
                query,
                key.transpose(1, 2),
            ) / math.sqrt(float(query.shape[-1]))
            output = output + torch.tanh(self.dynamic_channel_scale) * dynamic
        return output.to(device=device, dtype=dtype)

__all__ = [
    "MaskedChannelDistributionHead",
    "TypedChannelDistributionHead",
]

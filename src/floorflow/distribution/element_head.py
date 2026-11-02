"""Hierarchical semantic-element query distribution head."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.distribution.element_parents import parent_map
from floorflow.distribution.element_query_geometry import (
    mask_diagnostics,
    query_anchor_maps,
)
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.parent_gate import ParentReliabilityGate
from floorflow.distribution.query_budget import allocate_element_query_budgets
from floorflow.distribution.style_conditioning import (
    LowRankStyleAdapter,
    StyleElementResidual,
)
from floorflow.distribution.support_gate import apply_support_gate
from floorflow.model.official_mita_route import OfficialMiTARouteAttention


def _spatial_query_encoding(count: int, dimensions: int) -> torch.Tensor:
    """Deterministic coordinate prompts that discourage query collapse."""

    encoding = torch.zeros(count, dimensions)
    columns = max(1, int(count**0.5))
    rows = max(1, (count + columns - 1) // columns)
    frequencies = max(1, dimensions // 6)
    for index in range(count):
        x = 2.0 * (index % columns + 0.5) / columns - 1.0
        y = 2.0 * (index // columns + 0.5) / rows - 1.0
        for frequency in range(frequencies):
            channel = 6 * frequency
            if channel + 5 >= dimensions:
                break
            angle = (frequency + 1) * math.pi
            encoding[index, channel] = x
            encoding[index, channel + 1] = y
            encoding[index, channel + 2] = math.sin(angle * x / 2.0)
            encoding[index, channel + 3] = math.cos(angle * x / 2.0)
            encoding[index, channel + 4] = math.sin(angle * y / 2.0)
            encoding[index, channel + 5] = math.cos(angle * y / 2.0)
    return encoding * 0.5


class SemanticElementDistributionHead(nn.Module):
    """Decode wall→opening→room→furniture element queries."""

    GROUP_ORDER = ("wall", "opening", "room", "furniture", "support")

    def __init__(
        self,
        in_channels: int,
        hierarchy: ChannelHierarchy,
        *,
        hidden_channels: int = 96,
        queries_per_group: Mapping[str, int] | None = None,
        token_grid: int = 16,
        top_k: int = 8,
        output_resolution: int = 64,
        mask_version: str = "v2",
        num_styles: int = 0,
        style_rank: int = 8,
        hierarchy_support_strength: float = 0.0,
        dynamic_query_budget: bool = False,
    ) -> None:
        super().__init__()
        if mask_version not in {"v1", "v2"}:
            raise ValueError("mask_version must be v1 or v2")
        self.hierarchy = hierarchy
        self.token_grid = int(token_grid)
        self.top_k = int(top_k)
        self.output_resolution = int(output_resolution)
        self.mask_version = mask_version
        self.hierarchy_support_strength = float(hierarchy_support_strength)
        self.dynamic_query_budget = bool(dynamic_query_budget)
        self.style_adapter = (
            LowRankStyleAdapter(in_channels, num_styles, style_rank)
            if num_styles > 0
            else None
        )
        self.query_counts = {
            name: int((queries_per_group or {}).get(name, 8))
            for name in self.GROUP_ORDER
        }
        self.parent_reliability = ParentReliabilityGate(self.GROUP_ORDER)
        self.memory_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )
        self.group_queries = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.randn(count, hidden_channels) * 0.02
                )
                for name, count in self.query_counts.items()
            }
        )
        self.routes = nn.ModuleDict(
            {
                name: OfficialMiTARouteAttention(
                    hidden_channels,
                    num_heads=4,
                    router_count=min(16, max(1, count)),
                    top_k=self.top_k,
                )
                for name, count in self.query_counts.items()
            }
        )
        self.token_norm = nn.LayerNorm(hidden_channels)
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.mask_feature_decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )
        self.pixel_mask_projection = nn.Conv2d(
            hierarchy.num_classes,
            hidden_channels,
            1,
        )
        nn.init.zeros_(self.pixel_mask_projection.weight)
        nn.init.zeros_(self.pixel_mask_projection.bias)
        self.coordinate_mask_projection = nn.Conv2d(2, hidden_channels, 1)
        nn.init.normal_(self.coordinate_mask_projection.weight, std=0.02)
        nn.init.zeros_(self.coordinate_mask_projection.bias)
        self.query_anchor_logits = nn.ParameterDict(
            {
                name: nn.Parameter(query_anchor_maps(count, self.token_grid))
                for name, count in self.query_counts.items()
            }
        )
        self.query_anchor_scale = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(1.0))
                for name in self.GROUP_ORDER
            }
        )
        self.semantic_mask_scale = nn.ParameterDict(
            {
                "wall": nn.Parameter(torch.tensor(1.5)),
                "opening": nn.Parameter(torch.tensor(0.5)),
                "room": nn.Parameter(torch.tensor(1.0)),
                "furniture": nn.Parameter(torch.tensor(0.0)),
                "support": nn.Parameter(torch.tensor(0.0)),
            }
        )
        self.mask_scale = nn.Parameter(torch.ones(1))
        self.mask_bias = nn.Parameter(torch.zeros(1))
        self.mask_logit_scale = nn.Parameter(torch.tensor(4.0))
        self.mask_embedding_norm = nn.LayerNorm(hidden_channels)
        self.mask_feature_norm = nn.LayerNorm(hidden_channels)
        self.query_positions = nn.ParameterDict(
            {
                name: nn.Parameter(
                    _spatial_query_encoding(count, hidden_channels)
                )
                for name, count in self.query_counts.items()
            }
        )
        self.query_mask_bias = nn.ParameterDict(
            {
                name: nn.Parameter(torch.full((count,), -2.0))
                for name, count in self.query_counts.items()
            }
        )
        self.parent_logit_scale = nn.ParameterDict(
            {
                name: nn.Parameter(torch.ones(1))
                for name in self.GROUP_ORDER
            }
        )
        self.presence_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_channels, 1)
                for name in self.GROUP_ORDER
            }
        )
        self.type_heads = nn.ModuleDict(
            {
                name: nn.Linear(
                    hidden_channels,
                    len(self.hierarchy.groups[name]) + 1,
                )
                for name in self.GROUP_ORDER
            }
        )
        self.style_element_residual = (
            StyleElementResidual(
                num_styles,
                hidden_channels,
                {
                    name: len(self.hierarchy.groups[name]) + 1
                    for name in self.GROUP_ORDER
                },
                self.query_counts,
            )
            if num_styles > 0
            else None
        )

    def _parent_map(
        self,
        name: str,
        family_gates: Mapping[str, torch.Tensor],
        reference: torch.Tensor,
        room_parent_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return parent_map(
            name,
            family_gates,
            reference,
            room_parent_gate=room_parent_gate,
        )

    def _decode_v2_masks(
        self,
        name: str,
        mask_embedding: torch.Tensor,
        memory: torch.Tensor,
        parent: torch.Tensor,
        footprint: torch.Tensor,
        class_probability: torch.Tensor,
        type_logits: torch.Tensor,
    ) -> torch.Tensor:
        size = (self.output_resolution, self.output_resolution)
        mask_memory = F.adaptive_avg_pool2d(memory, size)
        mask_memory = self.mask_feature_decoder(mask_memory)
        mask_features = self.mask_feature_norm(
            mask_memory.flatten(2).transpose(1, 2)
        )
        mask_features = F.normalize(mask_features, dim=-1)
        mask_features = mask_features - mask_features.mean(dim=-2, keepdim=True)
        embedding = self.mask_embedding_norm(
            mask_embedding + 0.5 * self.query_positions[name].to(mask_embedding.dtype)
        )
        embedding = F.normalize(embedding, dim=-1)
        embedding = embedding - embedding.mean(dim=1, keepdim=True)
        masks = torch.einsum("bqd,bnd->bqn", embedding, mask_features).reshape(
            mask_embedding.shape[0],
            mask_embedding.shape[1],
            *size,
        )
        scale = F.softplus(self.mask_logit_scale).to(masks.dtype)
        masks = masks * scale
        anchors = F.interpolate(
            self.query_anchor_logits[name].to(masks.dtype)[None],
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        masks = masks + self.query_anchor_scale[name].to(masks.dtype) * anchors
        masks = masks + self.query_mask_bias[name].to(masks.dtype)[None, :, None, None]
        indices = self.hierarchy.groups[name]
        if indices:
            family_probability = F.interpolate(
                class_probability[:, indices].float(),
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            query_class = type_logits.softmax(dim=-1)[
                ..., : len(indices)
            ].to(family_probability.dtype)
            query_class = query_class / query_class.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0e-6)
            support = torch.einsum(
                "bqk,bkhw->bqhw",
                query_class,
                family_probability,
            )
            support_logits = torch.logit(
                support.clamp(1.0e-4, 1.0 - 1.0e-4)
            )
            masks = masks + self.semantic_mask_scale[name].to(
                masks.dtype
            ) * support_logits
        parent_weight = F.interpolate(
            parent,
            size=size,
            mode="bilinear",
            align_corners=False,
        ).clamp_min(1.0e-4)
        masks = masks + self.parent_logit_scale[name].to(masks.dtype) * parent_weight.log()
        if name in {"opening", "furniture", "support"}:
            masks = apply_support_gate(
                masks,
                parent,
                footprint,
                strength=self.hierarchy_support_strength,
            )
        output_footprint = F.interpolate(
            footprint.float(),
            size=size,
            mode="nearest",
        ).bool()
        return masks.masked_fill(~output_footprint, -20.0)

    def forward(
        self,
        features: torch.Tensor,
        pixel_distribution: dict[str, torch.Tensor],
        footprint: torch.Tensor | None = None,
        style_ids: torch.Tensor | None = None,
        prior_tokens: Mapping[str, torch.Tensor] | None = None,
        room_parent_gate: torch.Tensor | None = None,
    ) -> dict[str, object]:
        if footprint is None:
            footprint = torch.ones(
                features.shape[0],
                1,
                *features.shape[-2:],
                device=features.device,
                dtype=features.dtype,
            )
        elif footprint.ndim == 3:
            footprint = footprint[:, None]
        if self.style_adapter is not None:
            features = self.style_adapter(features, style_ids)
        projected_memory = self.memory_projection(features.float())
        if self.mask_version == "v2" and "probability" in pixel_distribution:
            projected_memory = projected_memory + self.pixel_mask_projection(
                pixel_distribution["probability"].float()
            )
            rows = torch.linspace(
                -1.0,
                1.0,
                features.shape[-2],
                device=features.device,
                dtype=features.dtype,
            )
            columns = torch.linspace(
                -1.0,
                1.0,
                features.shape[-1],
                device=features.device,
                dtype=features.dtype,
            )
            coordinates = torch.stack(
                torch.meshgrid(rows, columns, indexing="ij"),
                dim=0,
            )[None]
            projected_memory = projected_memory + self.coordinate_mask_projection(
                coordinates.to(projected_memory.dtype)
            )
        memory = F.adaptive_avg_pool2d(
            projected_memory,
            (self.token_grid, self.token_grid),
        )
        pixel_tokens = memory.flatten(2).transpose(1, 2)
        family_gates = pixel_distribution.get("family_gates", {})
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        element_tokens: dict[str, torch.Tensor] = {}
        query_budgets = (
            allocate_element_query_budgets(
                family_gates,
                features,
                self.query_counts,
            )
            if self.dynamic_query_budget
            else {
                name: torch.full(
                    (features.shape[0],),
                    count,
                    device=features.device,
                    dtype=torch.long,
                )
                for name, count in self.query_counts.items()
            }
        )
        parent_relations = {"wall": (), "opening": ("wall",), "room": ("wall", "opening"), "furniture": ("room",), "support": ("room",)}
        for name in self.GROUP_ORDER:
            raw_parent = self._parent_map(
                name,
                family_gates,
                features,
                room_parent_gate=room_parent_gate,
            )
            attention_parent, parent_diagnostics = self.parent_reliability(
                name,
                raw_parent,
                footprint,
            )
            parent_tokens_grid = F.adaptive_avg_pool2d(
                attention_parent,
                (self.token_grid, self.token_grid),
            ).flatten(2).transpose(1, 2)
            weighted_pixels = pixel_tokens * (
                0.25 + 0.75 * parent_tokens_grid.to(pixel_tokens.dtype)
            )
            direct_parent_tokens = [
                element_tokens[parent]
                for parent in parent_relations[name]
                if parent in element_tokens
            ]
            route_memory = torch.cat(
                [weighted_pixels, *direct_parent_tokens],
                dim=1,
            )
            queries = self.group_queries[name].to(route_memory.dtype)
            if self.mask_version == "v2":
                queries = queries + self.query_positions[name].to(route_memory.dtype)
            queries = queries[None].expand(
                features.shape[0],
                -1,
                -1,
            )
            if prior_tokens is not None and name in prior_tokens:
                queries = queries + prior_tokens[name].to(queries.dtype)
            tokens, route_diagnostics = self.routes[name](
                queries,
                route_memory,
            )
            tokens = self.token_norm(tokens)
            budget = query_budgets[name]
            active_queries = (
                torch.arange(
                    tokens.shape[1],
                    device=tokens.device,
                )[None] < budget[:, None]
            )
            if self.style_element_residual is not None:
                tokens = self.style_element_residual.augment_tokens(
                    tokens,
                    style_ids,
                )
            mask_embedding = self.mask_head(tokens)
            type_logits = self.type_heads[name](tokens)
            presence_logits = self.presence_heads[name](tokens).squeeze(-1)
            presence_logits = presence_logits.masked_fill(
                ~active_queries,
                -20.0,
            )
            if self.style_element_residual is not None:
                presence_logits, type_logits = (
                    self.style_element_residual.adjust_logits(
                        name,
                        presence_logits,
                        type_logits,
                        style_ids,
                    )
                )
            if self.mask_version == "v2":
                masks = self._decode_v2_masks(
                    name,
                    mask_embedding,
                    projected_memory,
                    raw_parent,
                    footprint,
                    pixel_distribution["probability"],
                    type_logits,
                )
            else:
                token_memory = self.token_norm(pixel_tokens)
                masks = torch.einsum(
                    "bqd,bnd->bqn",
                    mask_embedding,
                    token_memory,
                ).reshape(
                    features.shape[0],
                    tokens.shape[1],
                    self.token_grid,
                    self.token_grid,
                )
                masks = masks * self.mask_scale.to(masks.dtype) + self.mask_bias.to(
                    masks.dtype
                )
                mask_size = (
                    self.output_resolution,
                    self.output_resolution,
                )
                masks = F.interpolate(
                    masks,
                    size=mask_size,
                    mode="bilinear",
                    align_corners=False,
                )
                output_footprint = F.interpolate(
                    footprint.float(),
                    size=mask_size,
                    mode="nearest",
                ).bool()
                masks = masks.masked_fill(~output_footprint, -20.0)
            mask_size = (
                self.output_resolution,
                self.output_resolution,
            )
            parent_weight = F.interpolate(
                raw_parent,
                size=mask_size,
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            parent_gate = (
                masks.sigmoid() * parent_weight
            ).mean(dim=(2, 3))
            raw_parent_weight = F.interpolate(
                raw_parent,
                size=mask_size,
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            outputs[name] = {
                "tokens": tokens,
                "mask_logits": masks,
                "presence_logits": presence_logits,
                "type_logits": type_logits,
                "parent_gate": parent_gate,
                "parent_attention_gate": (
                    masks.sigmoid()
                    * F.interpolate(
                        attention_parent,
                        size=mask_size,
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0.0, 1.0)
                ).mean(dim=(2, 3)),
                "parent_raw_gate": (
                    masks.sigmoid() * raw_parent_weight
                ).mean(dim=(2, 3)),
                "parent_reliability_coverage": parent_diagnostics["coverage"],
                "parent_reliability_blend": parent_diagnostics["blend"],
                "parent_reliability_effect": parent_diagnostics["effect"],
                "routing_entropy": route_diagnostics["routing_entropy"],
                "effective_expert_fraction": route_diagnostics[
                    "effective_expert_fraction"
                ],
                "query_budget_mean": budget.float().mean(),
                "active_query_fraction": active_queries.float().mean(),
                **mask_diagnostics(masks),
            }
            element_tokens[name] = tokens
        return outputs


__all__ = ["SemanticElementDistributionHead"]

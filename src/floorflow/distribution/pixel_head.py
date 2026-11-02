"""Hierarchical pixel-class distribution with parent-conditioned attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.style_conditioning import (
    LowRankStyleAdapter,
    StylePixelResidual,
)
from floorflow.heads.pixel_distribution import PixelDistributionHead
from floorflow.model.official_mita_route import OfficialMiTARouteAttention


class HierarchicalPixelDistributionHead(nn.Module):
    """Couple wall→opening→room→furniture pixel probabilities."""

    def __init__(
        self,
        in_channels: int,
        hierarchy: ChannelHierarchy,
        hidden_channels: int = 96,
        token_grid: int = 16,
        num_styles: int = 0,
        style_rank: int = 8,
        dynamic_head_variant: str = "generated",
    ) -> None:
        super().__init__()
        self.hierarchy = hierarchy
        self.token_grid = int(token_grid)
        self.style_adapter = (
            LowRankStyleAdapter(in_channels, num_styles, style_rank)
            if num_styles > 0
            else None
        )
        self.style_pixel_residual = (
            StylePixelResidual(num_styles, hierarchy.num_classes - 1)
            if num_styles > 0
            else None
        )
        self.base = PixelDistributionHead(
            in_channels,
            hierarchy.num_classes,
            hidden_channels=hidden_channels,
            room_ids=hierarchy.room_ids,
            dynamic_head_variant=dynamic_head_variant,
        )
        self.memory_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )
        self.family_queries = nn.Parameter(
            torch.randn(5, hidden_channels) * 0.02
        )
        self.route = OfficialMiTARouteAttention(
            hidden_channels,
            num_heads=4,
            router_count=5,
            top_k=8,
        )
        self.feedback_route = OfficialMiTARouteAttention(
            hidden_channels,
            num_heads=4,
            router_count=5,
            top_k=5,
        )
        self.feedback_norm = nn.LayerNorm(hidden_channels)
        self.feedback_head = nn.Conv2d(
            hidden_channels,
            hierarchy.num_classes,
            1,
        )
        nn.init.zeros_(self.feedback_head.weight)
        nn.init.zeros_(self.feedback_head.bias)
        self.context_bias = nn.Linear(hidden_channels, 1)
        self.context_scale = nn.Parameter(torch.zeros(5))
        self.opening_parent_scale = nn.Parameter(torch.zeros(1))
        self.room_parent_scale = nn.Parameter(torch.zeros(1))
        self.furniture_parent_scale = nn.Parameter(torch.zeros(1))
        self.support_parent_scale = nn.Parameter(torch.zeros(1))

    def _group_sum(
        self,
        probability: torch.Tensor,
        indices: tuple[int, ...],
    ) -> torch.Tensor:
        if not indices:
            return probability.new_zeros(probability[:, :1])
        return probability[:, indices].sum(dim=1, keepdim=True)

    def _add_group_bias(
        self,
        logits: torch.Tensor,
        context: torch.Tensor,
        indices: tuple[int, ...],
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if not indices:
            return logits
        bias = self.context_bias(context)[..., None, None].to(logits.dtype)
        logits = logits.clone()
        logits[:, indices] = logits[:, indices] + scale.to(logits.dtype) * bias
        return logits

    def forward(
        self,
        features: torch.Tensor,
        style_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.style_adapter is not None:
            features = self.style_adapter(features, style_ids)
        output = dict(self.base(features))
        logits = output["logits"]
        memory = F.adaptive_avg_pool2d(
            self.memory_projection(features.float()),
            (self.token_grid, self.token_grid),
        ).flatten(2).transpose(1, 2)
        queries = self.family_queries.to(memory.dtype)[None].expand(
            features.shape[0],
            -1,
            -1,
        )
        context, diagnostics = self.route(queries, memory)
        feedback_tokens, feedback_diagnostics = self.feedback_route(
            memory,
            context,
        )
        feedback_tokens = self.feedback_norm(feedback_tokens)
        feedback_map = (
            feedback_tokens.transpose(1, 2)
            .reshape(
                features.shape[0],
                -1,
                self.token_grid,
                self.token_grid,
            )
        )
        feedback_map = F.interpolate(
            feedback_map,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        logits = logits + self.feedback_head(feedback_map).to(logits.dtype)
        groups = self.hierarchy.groups
        for index, name in enumerate(
            ("wall", "opening", "room", "furniture", "support")
        ):
            logits = self._add_group_bias(
                logits,
                context[:, index],
                groups[name],
                self.context_scale[index],
            )

        current_probability = logits.softmax(dim=1)
        wall_probability = self._group_sum(
            current_probability,
            groups["wall"],
        )
        room_probability = self._group_sum(
            current_probability,
            groups["room"],
        )
        opening_parent = F.max_pool2d(
            wall_probability,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        room_parent = (1.0 - wall_probability).clamp(0.0, 1.0)
        if groups["opening"]:
            logits[:, groups["opening"]] = (
                logits[:, groups["opening"]]
                + self.opening_parent_scale.to(logits.dtype)
                * opening_parent.clamp_min(1.0e-6).log()
                .to(logits.dtype)
            )
        if groups["room"]:
            logits[:, groups["room"]] = (
                logits[:, groups["room"]]
                + self.room_parent_scale.to(logits.dtype)
                * room_parent.clamp_min(1.0e-6).log()
                .to(logits.dtype)
            )
        if groups["furniture"]:
            logits[:, groups["furniture"]] = (
                logits[:, groups["furniture"]]
                + self.furniture_parent_scale.to(logits.dtype)
                * room_probability.clamp_min(1.0e-6).log()
                .to(logits.dtype)
            )
        if groups["support"]:
            logits[:, groups["support"]] = (
                logits[:, groups["support"]]
                + self.support_parent_scale.to(logits.dtype)
                * room_probability.clamp_min(1.0e-6).log()
                .to(logits.dtype)
            )
        if self.style_pixel_residual is not None:
            logits = self.style_pixel_residual(logits, style_ids)
        probability = logits.softmax(dim=1)
        entropy = -(
            probability.clamp_min(1.0e-8)
            * probability.clamp_min(1.0e-8).log()
        ).sum(dim=1, keepdim=True)
        reliability = 1.0 - entropy / max(
            1.0e-6,
            math.log(float(self.hierarchy.num_classes)),
        )
        room_indices = groups["room"]
        room_logits = (
            logits[:, room_indices]
            if room_indices
            else logits.new_zeros((logits.shape[0], 0, *logits.shape[-2:]))
        )
        return {
            **output,
            "logits": logits,
            "probability": probability,
            "room_logits": room_logits,
            "reliability": reliability,
            "family_tokens": context,
            "family_gates": {
                name: probability[:, indices].sum(dim=1, keepdim=True)
                for name, indices in groups.items()
                if indices
            },
            "routing_entropy": (
                diagnostics["routing_entropy"]
                + feedback_diagnostics["routing_entropy"]
            )
            / 2.0,
            "effective_expert_fraction": diagnostics[
                "effective_expert_fraction"
            ],
            "feedback_effective_expert_fraction": feedback_diagnostics[
                "effective_expert_fraction"
            ],
        }


__all__ = ["HierarchicalPixelDistributionHead"]

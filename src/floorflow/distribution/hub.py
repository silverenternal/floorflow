"""Unified hierarchical pixel and semantic-element distribution hub."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from floorflow.distribution.coupling_loss import (
    distribution_coupling_loss,
    element_to_pixel_probability,
)
from floorflow.distribution.element_head import SemanticElementDistributionHead
from floorflow.distribution.element_parents import family_parent_gates
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.pixel_head import HierarchicalPixelDistributionHead
from floorflow.model.footprint_cross_channel import FootprintCrossChannelPrior


class CoupledDistributionHub(nn.Module):
    """Coordinate dense pixel probabilities and element instance distributions."""

    def __init__(
        self,
        in_channels: int,
        hierarchy: ChannelHierarchy,
        *,
        hidden_channels: int = 96,
        pixel_grid: int = 16,
        element_grid: int = 16,
        element_resolution: int = 64,
        queries_per_group: Mapping[str, int] | None = None,
        element_mask_version: str = "v2",
        num_styles: int = 0,
        style_rank: int = 8,
        hierarchy_support_strength: float = 0.0,
        element_feedback_init: float = 0.0,
        dynamic_head_variant: str = "generated",
        dynamic_element_queries: bool = False,
    ) -> None:
        super().__init__()
        self.hierarchy = hierarchy
        self.cross_channel_prior = FootprintCrossChannelPrior(
            in_channels,
            hidden_channels,
            queries_per_group
            or {
                "wall": 8,
                "opening": 8,
                "room": 8,
                "furniture": 8,
                "support": 4,
            },
            token_grid=element_grid,
        )
        self.pixel = HierarchicalPixelDistributionHead(
            in_channels,
            hierarchy,
            hidden_channels=hidden_channels,
            token_grid=pixel_grid,
            num_styles=num_styles,
            style_rank=style_rank,
            dynamic_head_variant=dynamic_head_variant,
        )
        self.element = SemanticElementDistributionHead(
            in_channels,
            hierarchy,
            hidden_channels=hidden_channels,
            queries_per_group=queries_per_group,
            token_grid=element_grid,
            output_resolution=element_resolution,
            mask_version=element_mask_version,
            num_styles=num_styles,
            style_rank=style_rank,
            hierarchy_support_strength=hierarchy_support_strength,
            dynamic_query_budget=dynamic_element_queries,
        )
        self.element_feedback_scale = nn.Parameter(
            torch.full(
                (hierarchy.num_classes - 1,),
                float(element_feedback_init),
            )
        )

    def _refresh_pixel_output(
        self,
        pixel: dict[str, torch.Tensor],
        element_probability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feedback_logits = self.element_feedback_scale.to(
            pixel["logits"].dtype
        )[None, :, None, None] * element_probability.clamp_min(1.0e-6).log()
        logits = pixel["logits"].clone()
        logits[:, 1:] = logits[:, 1:] + feedback_logits
        probability = logits.softmax(dim=1)
        room_indices = self.hierarchy.groups["room"]
        room_logits = (
            logits[:, room_indices]
            if room_indices
            else logits.new_zeros((logits.shape[0], 0, *logits.shape[-2:]))
        )
        entropy = -(
            probability.clamp_min(1.0e-8)
            * probability.clamp_min(1.0e-8).log()
        ).sum(dim=1, keepdim=True)
        reliability = 1.0 - entropy / max(
            1.0e-6,
            math.log(float(self.hierarchy.num_classes)),
        )
        family_gates = {
            name: probability[:, indices].sum(dim=1, keepdim=True)
            for name, indices in self.hierarchy.groups.items()
            if indices
        }
        return {
            **pixel,
            "logits": logits,
            "probability": probability,
            "room_logits": room_logits,
            "reliability": reliability,
            "family_gates": family_gates,
            "element_feedback_probability": element_probability,
        }

    def forward(
        self,
        features: torch.Tensor,
        footprint: torch.Tensor | None = None,
        style_ids: torch.Tensor | None = None,
    ) -> dict[str, object]:
        pixel_base = self.pixel(features, style_ids=style_ids)
        parent_gates = family_parent_gates(pixel_base["family_gates"])
        cross_channel = (
            self.cross_channel_prior(features, footprint)
            if footprint is not None
            else {
                "tokens": None,
                "diagnostics": {},
            }
        )
        elements = self.element(
            features,
            pixel_base,
            footprint=footprint,
            style_ids=style_ids,
            prior_tokens=cross_channel["tokens"],
            room_parent_gate=parent_gates.get("room_region"),
        )
        element_probability = element_to_pixel_probability(
            elements,
            self.hierarchy,
            pixel_base["logits"],
        )
        pixel = self._refresh_pixel_output(
            pixel_base,
            element_probability,
        )
        return {
            "pixel": pixel,
            "pixel_base": pixel_base,
            "elements": elements,
            "family_gates": pixel["family_gates"],
            "parent_gates": parent_gates,
            "cross_channel_prior": cross_channel,
        }

    def loss(
        self,
        features: torch.Tensor,
        footprint: torch.Tensor | None = None,
        style_ids: torch.Tensor | None = None,
    ) -> tuple[dict[str, object], torch.Tensor, dict[str, torch.Tensor]]:
        output = self.forward(
            features,
            footprint=footprint,
            style_ids=style_ids,
        )
        coupling, components = distribution_coupling_loss(
            output["pixel"],
            output["elements"],
            self.hierarchy,
            footprint=footprint,
        )
        return output, coupling, components


__all__ = ["CoupledDistributionHub"]

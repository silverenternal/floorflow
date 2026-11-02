"""Joint backbone wrapper for distributions and contour-conditioned CFM."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from floorflow.distribution.cfm_bridge import DistributionCFMBridge
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.hub import CoupledDistributionHub
from floorflow.distribution.style_conditioning import LowRankStyleAdapter
from floorflow.distribution.velocity_refiner import SourceAwareVelocityRefiner


class DistributionCFMModel(nn.Module):
    """Share one contour backbone between velocity and distribution heads."""

    def __init__(
        self,
        base: nn.Module,
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
        velocity_refiner_hidden: int = 64,
        hierarchy_support_strength: float = 0.0,
        element_blend_init: float = 0.0,
        bridge_mode: str = "blend",
        element_feedback_init: float = 0.0,
        dynamic_head_variant: str = "generated",
        dynamic_element_queries: bool = False,
    ) -> None:
        super().__init__()
        if base.target_channels != hierarchy.num_classes - 1:
            raise ValueError("target channels must equal foreground class count")
        self.base = base
        self.target_channels = base.target_channels
        self.output = base.output
        self.hidden_channels = int(hidden_channels)
        self.pixel_grid = int(pixel_grid)
        self.element_grid = int(element_grid)
        self.element_resolution = int(element_resolution)
        self.element_mask_version = str(element_mask_version)
        self.num_styles = int(num_styles)
        self.style_rank = int(style_rank)
        self.velocity_refiner_hidden = int(velocity_refiner_hidden)
        self.hierarchy_support_strength = float(
            hierarchy_support_strength
        )
        self.element_blend_init = float(element_blend_init)
        self.bridge_mode = str(bridge_mode)
        self.element_feedback_init = float(element_feedback_init)
        self.dynamic_head_variant = str(dynamic_head_variant)
        self.dynamic_element_queries = bool(dynamic_element_queries)
        self.queries_per_group = dict(
            queries_per_group
            or {
                "wall": 8,
                "opening": 8,
                "room": 8,
                "furniture": 8,
                "support": 4,
            }
        )
        self.hub = CoupledDistributionHub(
            base.output.in_channels,
            hierarchy,
            hidden_channels=self.hidden_channels,
            pixel_grid=self.pixel_grid,
            element_grid=self.element_grid,
            queries_per_group=self.queries_per_group,
            element_resolution=self.element_resolution,
            element_mask_version=self.element_mask_version,
            num_styles=self.num_styles,
            style_rank=self.style_rank,
            hierarchy_support_strength=self.hierarchy_support_strength,
            element_feedback_init=self.element_feedback_init,
            dynamic_head_variant=self.dynamic_head_variant,
            dynamic_element_queries=self.dynamic_element_queries,
        )
        self.cfm_bridge = DistributionCFMBridge(
            hierarchy,
            element_blend_init=self.element_blend_init,
            bridge_mode=self.bridge_mode,
        )
        self.velocity_style_adapter = (
            LowRankStyleAdapter(
                base.output.in_channels,
                self.num_styles,
                self.style_rank,
            )
            if self.num_styles > 0
            else None
        )
        self.velocity_refiner = SourceAwareVelocityRefiner(
            base.output.in_channels,
            self.target_channels,
            self.velocity_refiner_hidden,
        )

    def _encode_features(
        self,
        x_t,
        time,
        footprint,
        floor_ids=None,
        condition_features=None,
    ):
        kwargs = {
            "floor_ids": floor_ids,
        }
        if condition_features is not None:
            kwargs["condition_features"] = condition_features
        return self.base._encode_features(x_t, time, footprint, **kwargs)

    def encode_condition(
        self,
        footprint: torch.Tensor,
        floor_ids: torch.Tensor | None = None,
    ):
        if not hasattr(self.base, "encode_condition"):
            return None
        time = footprint.new_zeros(footprint.shape[0])
        return self.base.encode_condition(
            footprint,
            time,
            floor_ids=floor_ids,
        )

    def _velocity_from_features(self, features, time=None):
        return self.base._velocity_from_features(features, time)

    def distribution_forward(
        self,
        features: torch.Tensor,
        footprint: torch.Tensor,
        style_ids: torch.Tensor | None = None,
    ) -> dict[str, object]:
        coupled = self.hub(
            features,
            footprint=footprint,
            style_ids=style_ids,
        )
        bridge = self.cfm_bridge(
            coupled["pixel"],
            coupled["elements"],
            footprint,
            element_probability=coupled["pixel"].get(
                "element_feedback_probability"
            ),
        )
        return {**coupled, "cfm": bridge}

    def contour_initial_state(
        self,
        footprint: torch.Tensor,
        floor_ids: torch.Tensor | None = None,
        style_ids: torch.Tensor | None = None,
        condition_features=None,
    ) -> dict[str, object]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        zero = footprint.new_zeros(
            footprint.shape[0],
            self.target_channels,
            *footprint.shape[-2:],
        )
        features = self._encode_features(
            zero,
            footprint.new_zeros(footprint.shape[0]),
            footprint,
            floor_ids=floor_ids,
            condition_features=condition_features,
        )
        partition_logits = None
        if getattr(self.base, "use_room_partition_head", False):
            room_prior = None
            if getattr(
                self.base,
                "use_room_partition_semantic_prior",
                False,
            ):
                semantic_logits = self.base.semantic_endpoint(features)
                room_prior = semantic_logits[
                    :,
                    self.base.room_class_ids.to(semantic_logits.device) - 1,
                ]
            partition_logits = self.base.room_partition(features, room_prior)
        return self.distribution_forward(
            features,
            footprint,
            style_ids=style_ids,
        ) | {
            "features": features,
            "room_partition_logits": partition_logits,
        }

    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        footprint: torch.Tensor,
        floor_ids: torch.Tensor | None = None,
        style_ids: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        condition_features=None,
    ) -> torch.Tensor:
        features = self._encode_features(
            x_t,
            time,
            footprint,
            floor_ids=floor_ids,
            condition_features=condition_features,
        )
        if self.velocity_style_adapter is not None:
            features = self.velocity_style_adapter(features, style_ids)
        velocity = self._velocity_from_features(features, time)
        if source is not None:
            velocity = velocity + self.velocity_refiner(
                features,
                x_t,
                source,
                footprint,
                time,
            )
        return velocity

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        footprint: torch.Tensor,
        floor_ids: torch.Tensor | None = None,
        style_ids: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        condition_features=None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        velocity = self.forward(
            x_t,
            time,
            footprint,
            floor_ids=floor_ids,
            style_ids=style_ids,
            source=source,
            condition_features=condition_features,
        )
        return velocity, {}


__all__ = ["DistributionCFMModel"]

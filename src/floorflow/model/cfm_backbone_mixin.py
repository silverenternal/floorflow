"""Reusable encode/decode methods for the contour CFM backbone."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.model.backbone_ops import (
    encode_condition_features,
    encode_features,
    velocity_from_features,
)


class CFMBackboneMixin(nn.Module):
    """Expose condition caching, feature encoding, and anchor operations."""

    def _encode_features(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        footprint: torch.Tensor,
        floor_ids: torch.Tensor | None = None,
        condition_features: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return encode_features(
            self,
            x_t,
            t,
            footprint,
            floor_ids=floor_ids,
            condition_features=condition_features,
        )

    def encode_condition(
        self,
        footprint: torch.Tensor,
        time: torch.Tensor,
        floor_ids: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return encode_condition_features(
            self,
            footprint,
            time,
            floor_ids,
        )

    def _velocity_from_features(
        self,
        features: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return velocity_from_features(self, features, t)

    def semantic_anchor_logits(
        self,
        reference: torch.Tensor,
        footprint: torch.Tensor,
        floor_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self.use_semantic_endpoint_head:
            return None
        zero = torch.zeros_like(reference)
        features = self._encode_features(
            zero,
            torch.zeros(
                reference.shape[0],
                device=reference.device,
                dtype=reference.dtype,
            ),
            footprint,
            floor_ids=floor_ids,
        )
        return self.semantic_endpoint(features)

    def fuse_semantic_anchor(
        self,
        flow_endpoint: torch.Tensor,
        anchor_logits: torch.Tensor | None,
        time_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_semantic_anchor_fusion or anchor_logits is None:
            return flow_endpoint, None
        return self.semantic_anchor_fusion(
            flow_endpoint,
            anchor_logits,
            time_value,
        )


__all__ = ["CFMBackboneMixin"]

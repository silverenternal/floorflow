"""Coordinate semantic channels through a full pixel-distribution head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.heads.pixel_distribution import PixelDistributionHead


class PixelDistributionController(nn.Module):
    """Keep the flow backbone intact and gate its semantic heads with a prior."""

    def __init__(
        self,
        base: nn.Module,
        *,
        num_classes: int,
        room_ids: tuple[int, ...],
        hidden_channels: int = 96,
        distribution_head: nn.Module | None = None,
        velocity_residual: bool = False,
        gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.base = base
        self.target_channels = getattr(base, "target_channels", num_classes - 1)
        self.output = getattr(base, "output", None)
        self.distribution = (
            distribution_head
            if distribution_head is not None
            else PixelDistributionHead(
                base.output.in_channels,
                num_classes,
                hidden_channels=hidden_channels,
                room_ids=room_ids,
            )
        )
        self.prior_scale = nn.Parameter(torch.zeros(1))
        feature_channels = int(base.output.in_channels)
        self.velocity_gate = (
            nn.Sequential(
                nn.Conv2d(feature_channels + 1, int(gate_hidden), 3, padding=1),
                nn.GroupNorm(max(1, int(gate_hidden) // 8), int(gate_hidden)),
                nn.SiLU(),
                nn.Conv2d(int(gate_hidden), self.target_channels, 1),
            )
            if velocity_residual
            else None
        )
        if self.velocity_gate is not None:
            nn.init.zeros_(self.velocity_gate[-1].weight)
            nn.init.zeros_(self.velocity_gate[-1].bias)

    def _encode_features(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        footprint: torch.Tensor,
        floor_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base._encode_features(
            x_t,
            time,
            footprint,
            floor_ids=floor_ids,
        )

    def _velocity_from_features(
        self,
        features: torch.Tensor,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base._velocity_from_features(features, time)

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        footprint: torch.Tensor,
        floor_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        features = self.base._encode_features(
            x_t,
            time,
            footprint,
            floor_ids=floor_ids,
        )
        velocity = self.base._velocity_from_features(features, time)
        distribution = self.distribution(features)
        if self.velocity_gate is not None:
            reliability = distribution["reliability"].to(features.dtype)
            gate = self.velocity_gate(
                torch.cat([features, reliability], dim=1)
            )
            probability = distribution["probability"].to(x_t.dtype)
            endpoint_residual = probability[:, 1:] - x_t
            if footprint.ndim == 3:
                footprint = footprint[:, None]
            velocity = (
                velocity
                + gate.to(velocity.dtype)
                * endpoint_residual
                * footprint.float()
            )
        _ignored_velocity, auxiliary = self.base.forward_with_aux(
            x_t,
            time,
            footprint,
            floor_ids=floor_ids,
        )
        auxiliary = dict(auxiliary)
        room_logits = auxiliary.get("room_partition_logits")
        if room_logits is not None:
            prior = F.interpolate(
                distribution["room_logits"],
                size=room_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if prior.shape[1] == room_logits.shape[1]:
                room_logits = room_logits + self.prior_scale * prior
                auxiliary["room_partition_logits"] = room_logits
                auxiliary["distribution_room_prior"] = prior
        objects = auxiliary.get("structured_objects")
        if isinstance(objects, dict):
            reliability = distribution["reliability"]
            gated = {}
            for name, value in objects.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim == 4
                    and name.endswith("_logits")
                ):
                    reliability_gate = F.interpolate(
                        reliability,
                        size=value.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    value = value + self.prior_scale * (reliability_gate - 0.5)
                gated[name] = value
            auxiliary["structured_objects"] = gated
        auxiliary["pixel_distribution"] = distribution
        if self.velocity_gate is not None:
            auxiliary["distribution_velocity_gate"] = gate
        return velocity, auxiliary


__all__ = ["PixelDistributionController"]

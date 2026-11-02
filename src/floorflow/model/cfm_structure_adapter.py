"""Structured family evidence and multi-scale CFM conditioning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.distribution.hierarchy import ChannelHierarchy


def _foreground_indices(class_ids: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(max(0, int(class_id) - 1) for class_id in class_ids)


class CfmFamilyStreamProjector(nn.Module):
    """Create independent room/wall/opening/furniture evidence streams."""

    STREAMS = ("room", "wall", "opening", "furniture")

    def __init__(
        self,
        target_channels: int,
        condition_channels: int,
        hidden_channels: int,
        groups: dict[str, tuple[int, ...]],
    ) -> None:
        super().__init__()
        self.target_channels = int(target_channels)
        self.hidden_channels = int(hidden_channels)
        self.groups = {
            name: _foreground_indices(groups.get(name, ()))
            for name in self.STREAMS
        }
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(2, hidden_channels, 1),
                    nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
                    nn.SiLU(),
                )
                for name in self.STREAMS
            }
        )
        self.condition_projection = nn.Sequential(
            nn.Conv2d(condition_channels, hidden_channels, 1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state.ndim != 4 or condition.ndim != 4:
            raise ValueError("state and condition must be [B,C,H,W]")
        size = condition.shape[-2:]
        state = F.adaptive_avg_pool2d(state.float(), size)
        output = {}
        for name in self.STREAMS:
            indices = self.groups[name]
            family = (
                state[:, indices]
                if indices
                else state.new_zeros(
                    state.shape[0],
                    1,
                    *state.shape[-2:],
                )
            )
            evidence = torch.cat(
                [
                    family.mean(dim=1, keepdim=True),
                    family.amax(dim=1, keepdim=True),
                ],
                dim=1,
            )
            output[name] = (
                (
                    self.projections[name](evidence)
                    + self.condition_projection(condition.float())
                )
                .flatten(2)
                .transpose(1, 2)
            )
        condition_tokens = (
            self.condition_projection(condition.float())
            .flatten(2)
            .transpose(1, 2)
        )
        output["condition"] = condition_tokens
        output["direct"] = condition_tokens
        return output


class CfmStructureTokenEncoder(nn.Module):
    """Build one global structural token from typed state and envelope."""

    def __init__(
        self,
        target_channels: int,
        hidden_dim: int,
        groups: dict[str, tuple[int, ...]],
    ) -> None:
        super().__init__()
        self.groups = {
            name: _foreground_indices(groups.get(name, ()))
            for name in CfmFamilyStreamProjector.STREAMS
        }
        self.projection = nn.Sequential(
            nn.Conv2d(len(self.groups) * 2 + 1, hidden_dim, 3, padding=1),
            nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(
        self,
        state: torch.Tensor,
        footprint: torch.Tensor,
    ) -> torch.Tensor:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        state = F.adaptive_avg_pool2d(state.float(), (8, 8))
        envelope = F.adaptive_avg_pool2d(footprint.float(), (8, 8))
        evidence = []
        for indices in self.groups.values():
            family = state[:, indices] if indices else state[:, :1] * 0.0
            evidence.extend(
                [
                    family.mean(dim=1, keepdim=True),
                    family.amax(dim=1, keepdim=True),
                ]
            )
        hidden = self.projection(torch.cat([*evidence, envelope], dim=1))
        return hidden.flatten(2).mean(dim=-1)


class FeatureFamilyStreamProjector(nn.Module):
    """Create independently projected family streams at the velocity scale."""

    STREAMS = ("room", "wall", "opening", "furniture", "condition")

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Conv2d(int(channels), int(channels), 1)
                for name in self.STREAMS
            }
        )

    def forward(
        self,
        feature: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if condition.shape[-2:] != feature.shape[-2:]:
            condition = F.interpolate(
                condition,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        streams = {}
        for name in ("room", "wall", "opening", "furniture"):
            streams[name] = (
                self.projections[name](feature).flatten(2).transpose(1, 2)
            )
        streams["condition"] = (
            self.projections["condition"](condition)
            .flatten(2)
            .transpose(1, 2)
        )
        return streams


class MultiScaleStructureInjector(nn.Module):
    """Inject the structural token at every U-Net scale as identity FiLM."""

    def __init__(
        self,
        channels: tuple[int, ...],
        token_dim: int,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(int(token_dim), int(channel) * 2) for channel in channels]
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        level: int,
        hidden: torch.Tensor,
        token: torch.Tensor,
    ) -> torch.Tensor:
        scale, bias = self.projections[level](token).chunk(2, dim=-1)
        return hidden * (1.0 + torch.tanh(scale)[..., None, None]) + torch.tanh(
            bias
        )[..., None, None]


def build_cfm_structure_modules(
    target_channels: int,
    condition_channels: int,
    channels: tuple[int, ...],
    taxonomy: FloorplanTaxonomy | None,
    *,
    typed_family_streams: bool,
    multiscale_structure_injection: bool,
) -> tuple[
    CfmFamilyStreamProjector | None,
    CfmStructureTokenEncoder | None,
    MultiScaleStructureInjector | None,
    FeatureFamilyStreamProjector | None,
]:
    family = token_encoder = injector = velocity_streams = None
    if taxonomy is not None and (
        typed_family_streams or multiscale_structure_injection
    ):
        groups = ChannelHierarchy.from_taxonomy(taxonomy).groups
        if typed_family_streams:
            family = CfmFamilyStreamProjector(
                target_channels,
                condition_channels,
                channels[-1],
                groups,
            )
            velocity_streams = FeatureFamilyStreamProjector(channels[0])
        if multiscale_structure_injection:
            token_encoder = CfmStructureTokenEncoder(
                target_channels,
                channels[-1],
                groups,
            )
            injector = MultiScaleStructureInjector(channels, channels[-1])
    return family, token_encoder, injector, velocity_streams


def enable_cfm_structure(
    base: nn.Module,
    taxonomy: FloorplanTaxonomy,
    *,
    static_condition_encoder: bool = True,
    typed_family_streams: bool = True,
    multiscale_structure_injection: bool = True,
) -> nn.Module:
    """Attach structured CFM modules to an existing contour backbone."""

    base.static_condition_encoder = bool(static_condition_encoder)
    base.typed_family_streams = bool(typed_family_streams)
    base.multiscale_structure_injection = bool(
        multiscale_structure_injection
    )
    base.condition_encoder.use_time = not bool(static_condition_encoder)
    base_channels = int(base.output_norm.num_channels)
    channels = tuple(
        base_channels * (2**level)
        for level in range(int(base.levels))
    )
    (
        base.family_stream_evidence,
        base.structure_token_encoder,
        base.structure_injector,
        base.velocity_family_streams,
    ) = build_cfm_structure_modules(
        int(base.target_channels),
        channels[-1],
        channels,
        taxonomy,
        typed_family_streams=typed_family_streams,
        multiscale_structure_injection=multiscale_structure_injection,
    )
    return base


__all__ = [
    "CfmFamilyStreamProjector",
    "CfmStructureTokenEncoder",
    "FeatureFamilyStreamProjector",
    "MultiScaleStructureInjector",
    "build_cfm_structure_modules",
    "enable_cfm_structure",
]

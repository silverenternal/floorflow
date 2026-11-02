"""Fuse predicted-control fields into typed spatial evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.heads.room_segmenter import ConvNormAct
from floorflow.heads.typed_source_spatial_router import TypedSourceSpatialRouter


class PredictedControlFieldEncoder(nn.Module):
    """Fuse room, structure, furniture, support, and program controls."""

    def __init__(
        self,
        control_channels: tuple[int, int, int, int],
        program_channels: int,
        *,
        grid: int = 64,
        hidden_channels: int = 48,
        evidence_context: bool = False,
        typed_source_spatial_router: bool = False,
    ) -> None:
        super().__init__()
        self.grid = int(grid)
        self.hidden_channels = int(hidden_channels)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(channels), self.hidden_channels, 1),
                    nn.GroupNorm(
                        max(1, self.hidden_channels // 16),
                        self.hidden_channels,
                    ),
                    nn.SiLU(),
                )
                for channels in control_channels
            ]
        )
        self.program = nn.Sequential(
            nn.Linear(int(program_channels), self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        self.envelope = nn.Sequential(
            nn.Conv2d(1, self.hidden_channels, 1),
            nn.GroupNorm(max(1, self.hidden_channels // 16), self.hidden_channels),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            ConvNormAct(self.hidden_channels * 5 + 1, self.hidden_channels),
            ConvNormAct(self.hidden_channels, self.hidden_channels),
        )
        self.evidence_fusion = nn.Sequential(
            nn.Linear(self.hidden_channels * 11, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        ) if evidence_context else None
        if self.evidence_fusion is not None:
            nn.init.normal_(
                self.evidence_fusion[-1].weight,
                mean=0.0,
                std=1.0e-3,
            )
            nn.init.zeros_(self.evidence_fusion[-1].bias)
        self.source_router = (
            TypedSourceSpatialRouter(
                self.hidden_channels,
                source_count=4,
            )
            if typed_source_spatial_router
            else None
        )
        self.last_source_routing: dict[str, torch.Tensor] | None = None

    def encode_details(
        self,
        controls: dict[str, torch.Tensor],
        footprint: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        fields = (
            controls["room_logits"],
            controls["structure"],
            controls["furniture"],
            controls["support"],
        )
        projected = []
        for field, projection in zip(fields, self.projections, strict=True):
            field = F.interpolate(
                field.float(),
                size=(self.grid, self.grid),
                mode="bilinear",
                align_corners=False,
            )
            projected.append(projection(field))
        program_evidence = self.program(controls["program"].float())
        program = program_evidence[..., None, None].expand(
            -1,
            -1,
            self.grid,
            self.grid,
        )
        footprint_field = F.interpolate(
            footprint.float()[:, None]
            if footprint.ndim == 3
            else footprint.float(),
            size=(self.grid, self.grid),
            mode="nearest",
        )
        envelope_features = self.envelope(footprint_field)
        features = self.fuse(
            torch.cat([*projected, program, footprint_field], dim=1)
        )
        self.last_source_routing = None
        if self.source_router is not None:
            features, routing = self.source_router(
                tuple(projected),
                features,
                footprint_field,
            )
            self.last_source_routing = routing
        evidence = None
        if self.evidence_fusion is not None:
            summaries = []
            for field in projected:
                summaries.extend(
                    [
                        field.mean(dim=(2, 3)),
                        field.amax(dim=(2, 3)),
                    ]
                )
            summaries.append(program_evidence)
            summaries.extend(
                [
                    envelope_features.mean(dim=(2, 3)),
                    envelope_features.amax(dim=(2, 3)),
                ]
            )
            evidence = self.evidence_fusion(torch.cat(summaries, dim=-1))
        source_tokens = torch.stack(
            [
                field.mean(dim=(2, 3)) + field.amax(dim=(2, 3))
                for field in projected
            ]
            + [
                program_evidence,
                envelope_features.mean(dim=(2, 3))
                + envelope_features.amax(dim=(2, 3)),
            ],
            dim=1,
        )
        return features, evidence, source_tokens

    def encode_with_evidence(
        self,
        controls: dict[str, torch.Tensor],
        footprint: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features, evidence, _source_tokens = self.encode_details(
            controls,
            footprint,
        )
        return features, evidence

    def forward(
        self,
        controls: dict[str, torch.Tensor],
        footprint: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_with_evidence(controls, footprint)[0]


__all__ = ["PredictedControlFieldEncoder"]

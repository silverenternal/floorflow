"""Compact boundary-conditioned encoder without a ResNet parameter floor."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.heads.room_segmenter import ConvNormAct, prepare_room_condition


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(max(1, channels // 16), channels),
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.network(value))


class CompactBoundaryEncoder(nn.Module):
    """A compact U-Net-style boundary encoder with a segmenter-compatible API."""

    def __init__(
        self,
        num_room_classes: int,
        *,
        decoder_channels: int = 96,
        channels: tuple[int, ...] = (64, 128, 256, 384),
        blocks: tuple[int, ...] = (2, 4, 6, 3),
    ) -> None:
        super().__init__()
        if len(channels) != len(blocks):
            raise ValueError("encoder channels and blocks must share length")
        if not channels:
            raise ValueError("encoder needs at least one stage")
        self.channels = tuple(int(value) for value in channels)
        self.blocks_per_stage = tuple(int(value) for value in blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.channels[0], 3, stride=2, padding=1),
            nn.GroupNorm(max(1, self.channels[0] // 16), self.channels[0]),
            nn.SiLU(),
        )
        stages = []
        current = self.channels[0]
        for stage_index, (channels_out, count) in enumerate(
            zip(self.channels, self.blocks_per_stage, strict=True)
        ):
            blocks_in_stage = []
            if stage_index:
                blocks_in_stage.append(
                    nn.Sequential(
                        nn.Conv2d(
                            current,
                            channels_out,
                            3,
                            stride=2,
                            padding=1,
                        ),
                        nn.GroupNorm(
                            max(1, channels_out // 16),
                            channels_out,
                        ),
                        nn.SiLU(),
                    )
                )
            blocks_in_stage.extend(
                ResidualBlock(channels_out)
                for _ in range(max(1, count))
            )
            stages.append(nn.Sequential(*blocks_in_stage))
            current = channels_out
        self.stages = nn.ModuleList(stages)
        self.reduce = nn.ModuleList(
            nn.Conv2d(channel, decoder_channels, 1)
            for channel in self.channels
        )
        self.decode_blocks = nn.ModuleList(
            ConvNormAct(decoder_channels * 2, decoder_channels)
            for _ in range(len(self.channels) - 1)
        )
        self.final = nn.Sequential(
            ConvNormAct(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, num_room_classes, 1),
        )

    def encode_control(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = (
            prepare_room_condition(footprint)
            if control_image is None
            else control_image.float()
        )
        if condition.shape[1] != 3:
            raise ValueError("compact encoder control input must have three channels")
        hidden = self.stem(condition)
        features = []
        for stage in self.stages:
            hidden = stage(hidden)
            features.append(hidden)
        decoded = self.reduce[-1](features[-1])
        for index in range(len(features) - 2, -1, -1):
            decoded = F.interpolate(
                decoded,
                size=features[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            projected = self.reduce[index](features[index])
            decoded = self.decode_blocks[index](
                torch.cat([decoded, projected], dim=1)
            )
        return F.interpolate(
            decoded,
            size=footprint.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.final(
            self.encode_control(footprint, control_image=control_image)
        )


__all__ = ["CompactBoundaryEncoder", "ResidualBlock"]

"""Contour-only room segmentation with a pretrained residual encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, out_channels // 16), out_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


@dataclass(frozen=True)
class RoomEncoderFeatures:
    """Spatial features exposed by the frozen room encoder."""

    hidden: torch.Tensor
    x1: torch.Tensor
    x2: torch.Tensor
    x3: torch.Tensor


def prepare_room_condition(footprint: torch.Tensor) -> torch.Tensor:
    """Build a stable three-channel representation of the exterior contour."""

    if footprint.ndim == 4:
        mask = footprint[:, 0]
    else:
        mask = footprint
    mask = (mask > 0.5).float()
    eroded = -F.max_pool2d(-mask[:, None], 5, stride=1, padding=2)[:, 0]
    boundary = (mask - eroded).clamp(0.0, 1.0)
    deep_interior = -F.max_pool2d(-mask[:, None], 15, stride=1, padding=7)[:, 0]
    channels = torch.stack([mask, deep_interior, boundary], dim=1)
    mean = channels.new_tensor([0.45, 0.45, 0.45])[None, :, None, None]
    std = channels.new_tensor([0.25, 0.25, 0.25])[None, :, None, None]
    return (channels - mean) / std


class ContourRoomSegmenter(nn.Module):
    """ResNet18 encoder with a lightweight full-resolution FPN decoder."""

    def __init__(
        self,
        num_room_classes: int,
        *,
        pretrained: bool = True,
        decoder_channels: int = 96,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        encoder = resnet18(weights=weights)
        self.stem = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
        )
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.reduce4 = nn.Conv2d(512, decoder_channels, 1)
        self.decode3 = ConvNormAct(decoder_channels + 256, decoder_channels)
        self.reduce3 = nn.Conv2d(decoder_channels, decoder_channels, 1)
        self.decode2 = ConvNormAct(decoder_channels + 128, decoder_channels)
        self.reduce2 = nn.Conv2d(decoder_channels, decoder_channels, 1)
        self.decode1 = ConvNormAct(decoder_channels + 64, decoder_channels)
        self.final = nn.Sequential(
            ConvNormAct(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, num_room_classes, 1),
        )

    def encode_control_features(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> RoomEncoderFeatures:
        condition = (
            prepare_room_condition(footprint)
            if control_image is None
            else control_image.float()
        )
        if condition.shape[1] != 3:
            raise ValueError("room segmenter control image must have three channels")
        x1 = self.layer1(self.stem(condition))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        hidden = self.reduce4(x4)
        hidden = F.interpolate(hidden, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.decode3(torch.cat([hidden, x3], dim=1))
        hidden = self.reduce3(hidden)
        hidden = F.interpolate(hidden, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.decode2(torch.cat([hidden, x2], dim=1))
        hidden = self.reduce2(hidden)
        hidden = F.interpolate(hidden, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.decode1(torch.cat([hidden, x1], dim=1))
        hidden = F.interpolate(hidden, size=footprint.shape[-2:], mode="bilinear", align_corners=False)
        return RoomEncoderFeatures(
            hidden=hidden,
            x1=x1,
            x2=x2,
            x3=x3,
        )

    def encode_control(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_control_features(
            footprint,
            control_image=control_image,
        ).hidden

    def forward(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict room logits from contour only or a normalized raster control."""

        return self.final(
            self.encode_control(footprint, control_image=control_image)
        )

"""ConvNeXt-Tiny room control head with the ContourRoomSegmenter contract."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from floorflow.heads.room_segmenter import ConvNormAct


class ConvNeXtRoomControl(nn.Module):
    def __init__(
        self,
        num_room_classes: int,
        *,
        pretrained: bool = True,
        decoder_channels: int = 128,
        backbone_weights: str | Path | None = None,
    ) -> None:
        super().__init__()
        if backbone_weights is not None and Path(backbone_weights).is_file():
            weights = None
        else:
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        if backbone_weights is not None and Path(backbone_weights).is_file():
            state = torch.load(
                backbone_weights,
                map_location="cpu",
                weights_only=True,
            )
            backbone.load_state_dict(state, strict=False)
        self.features = backbone.features
        hidden = int(decoder_channels)
        self.reduce4 = nn.Conv2d(768, hidden, 1)
        self.decode3 = ConvNormAct(hidden + 384, hidden)
        self.reduce3 = nn.Conv2d(hidden, hidden, 1)
        self.decode2 = ConvNormAct(hidden + 192, hidden)
        self.reduce2 = nn.Conv2d(hidden, hidden, 1)
        self.decode1 = ConvNormAct(hidden + 96, hidden)
        self.final = nn.Sequential(
            ConvNormAct(hidden, hidden),
            nn.Conv2d(hidden, int(num_room_classes), 1),
        )

    def encode_control(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if control_image is None:
            if footprint.ndim == 3:
                footprint = footprint[:, None]
            control_image = footprint.float().expand(-1, 3, -1, -1)
        x = self.features[0](control_image.float())
        x1 = self.features[1](x)
        x2 = self.features[3](self.features[2](x1))
        x3 = self.features[5](self.features[4](x2))
        x4 = self.features[7](self.features[6](x3))
        hidden = self.reduce4(x4)
        hidden = F.interpolate(
            hidden,
            size=x3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = self.decode3(torch.cat([hidden, x3], dim=1))
        hidden = self.reduce3(hidden)
        hidden = F.interpolate(
            hidden,
            size=x2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = self.decode2(torch.cat([hidden, x2], dim=1))
        hidden = self.reduce2(hidden)
        hidden = F.interpolate(
            hidden,
            size=x1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = self.decode1(torch.cat([hidden, x1], dim=1))
        return F.interpolate(
            hidden,
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


__all__ = ["ConvNeXtRoomControl"]

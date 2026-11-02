"""Contour and room-program conditioned coarse layout predictor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.heads.coarse_structure_refiner import _ResidualBlock


class ContourProgramLayoutPredictor(nn.Module):
    def __init__(
        self,
        num_room_classes: int,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        base = int(base_channels)
        input_channels = int(num_room_classes) + 1
        self.enc256 = _ResidualBlock(input_channels, base)
        self.enc128 = _ResidualBlock(base, base * 2)
        self.enc64 = _ResidualBlock(base * 2, base * 4)
        self.enc32 = _ResidualBlock(base * 4, base * 8)
        self.dec64 = _ResidualBlock(base * 12, base * 4)
        self.dec128 = _ResidualBlock(base * 6, base * 2)
        self.dec256 = _ResidualBlock(base * 3, base)
        self.out = nn.Conv2d(base, int(num_room_classes), 1)

    @staticmethod
    def _up(inputs, skip):
        return F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        footprint: torch.Tensor,
        program: torch.Tensor,
    ) -> torch.Tensor:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        program_map = program[:, :, None, None].to(footprint.dtype)
        program_map = program_map.expand(
            -1,
            -1,
            footprint.shape[-2],
            footprint.shape[-1],
        )
        inputs = torch.cat([footprint.float(), program_map], dim=1)
        skip256 = self.enc256(inputs)
        skip128 = self.enc128(F.avg_pool2d(skip256, 2))
        skip64 = self.enc64(F.avg_pool2d(skip128, 2))
        low = self.enc32(F.avg_pool2d(skip64, 2))
        hidden64 = self.dec64(
            torch.cat([self._up(low, skip64), skip64], dim=1)
        )
        hidden128 = self.dec128(
            torch.cat([self._up(hidden64, skip128), skip128], dim=1)
        )
        hidden256 = self.dec256(
            torch.cat([self._up(hidden128, skip256), skip256], dim=1)
        )
        return self.out(hidden256)


class ContourJointLayoutPredictor(nn.Module):
    """Predict room presence and coarse layout from contour in one pass."""

    def __init__(
        self,
        num_room_classes: int,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        base = int(base_channels)
        self.num_room_classes = int(num_room_classes)
        self.enc256 = _ResidualBlock(1, base)
        self.enc128 = _ResidualBlock(base, base * 2)
        self.enc64 = _ResidualBlock(base * 2, base * 4)
        self.enc32 = _ResidualBlock(base * 4, base * 8)
        self.program_head = nn.Sequential(
            nn.Linear(base * 16, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, self.num_room_classes),
        )
        self.program_projection = nn.Sequential(
            nn.Conv2d(self.num_room_classes, base * 8, 1),
            nn.SiLU(),
        )
        self.dec64 = _ResidualBlock(base * 12, base * 4)
        self.dec128 = _ResidualBlock(base * 6, base * 2)
        self.dec256 = _ResidualBlock(base * 3, base)
        self.layout_head = nn.Conv2d(base, self.num_room_classes, 1)

    @staticmethod
    def _up(inputs, skip):
        return F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, footprint: torch.Tensor) -> dict[str, torch.Tensor]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        skip256 = self.enc256(footprint.float())
        skip128 = self.enc128(F.avg_pool2d(skip256, 2))
        skip64 = self.enc64(F.avg_pool2d(skip128, 2))
        low = self.enc32(F.avg_pool2d(skip64, 2))
        pooled = torch.cat([low.mean(dim=(2, 3)), low.amax(dim=(2, 3))], dim=1)
        presence_logits = self.program_head(pooled)
        probability = presence_logits.sigmoid()
        program_map = probability[:, :, None, None].expand(
            -1,
            -1,
            low.shape[-2],
            low.shape[-1],
        )
        conditioned_low = low + self.program_projection(program_map)
        hidden64 = self.dec64(
            torch.cat([self._up(conditioned_low, skip64), skip64], dim=1)
        )
        hidden128 = self.dec128(
            torch.cat([self._up(hidden64, skip128), skip128], dim=1)
        )
        hidden256 = self.dec256(
            torch.cat([self._up(hidden128, skip256), skip256], dim=1)
        )
        return {
            "presence_logits": presence_logits,
            "layout_logits": self.layout_head(hidden256),
        }


__all__ = ["ContourJointLayoutPredictor", "ContourProgramLayoutPredictor"]

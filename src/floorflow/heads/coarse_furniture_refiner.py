"""Second-stage furniture refinement from coarse CAD layout control."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.coarse_structure_refiner import _ResidualBlock


def coarse_furniture_control(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    size: int = 128,
) -> torch.Tensor:
    room_ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=labels.device)
    rooms = torch.stack(
        [(labels == class_id).float() for class_id in room_ids],
        dim=1,
    )
    furniture = torch.stack(
        [(labels == class_id).float() for class_id in furniture_ids],
        dim=1,
    )
    footprint_channel = footprint.float()
    if footprint_channel.ndim == 3:
        footprint_channel = footprint_channel[:, None]
    control = torch.cat([rooms, furniture, footprint_channel], dim=1)
    return F.adaptive_max_pool2d(control, (int(size), int(size)))


class CoarseFurnitureRefiner(nn.Module):
    """Upsample coarse furniture support and predict per-class instances."""

    def __init__(
        self,
        input_channels: int,
        num_furniture_classes: int,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        base = int(base_channels)
        self.enc128 = _ResidualBlock(input_channels, base)
        self.enc64 = _ResidualBlock(base, base * 2)
        self.enc32 = _ResidualBlock(base * 2, base * 4)
        self.dec64 = _ResidualBlock(base * 6, base * 2)
        self.dec128 = _ResidualBlock(base * 3, base)
        self.dec256 = _ResidualBlock(base, base)
        self.foreground = nn.Conv2d(base, 1, 1)
        self.class_logits = nn.Conv2d(base, int(num_furniture_classes), 1)

    @staticmethod
    def _up(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        control: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        skip128 = self.enc128(control)
        skip64 = self.enc64(F.avg_pool2d(skip128, 2))
        low = self.enc32(F.avg_pool2d(skip64, 2))
        hidden64 = self.dec64(
            torch.cat([self._up(low, skip64), skip64], dim=1)
        )
        hidden128 = self.dec128(
            torch.cat([self._up(hidden64, skip128), skip128], dim=1)
        )
        hidden256 = self.dec256(
            F.interpolate(
                hidden128,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
        )
        if output_size is None:
            output_size = (
                control.shape[-2] * 2,
                control.shape[-1] * 2,
            )
        if hidden256.shape[-2:] != tuple(output_size):
            hidden256 = F.interpolate(
                hidden256,
                size=tuple(output_size),
                mode="bilinear",
                align_corners=False,
            )
        return {
            "foreground_logits": self.foreground(hidden256),
            "class_logits": self.class_logits(hidden256),
        }


__all__ = ["CoarseFurnitureRefiner", "coarse_furniture_control"]

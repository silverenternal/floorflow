"""Second-stage room-support refinement separate from visible semantics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.coarse_structure_refiner import _ResidualBlock


def coarse_support_control(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    size: int = 192,
) -> torch.Tensor:
    room_ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    rooms = torch.stack(
        [(labels == class_id).float() for class_id in room_ids],
        dim=1,
    )
    wall = (labels == taxonomy.class_id("wall")).float()[:, None]
    opening = torch.isin(
        labels,
        torch.tensor(taxonomy.opening_ids, device=labels.device),
    ).float()[:, None]
    furniture = torch.isin(
        labels,
        torch.tensor(taxonomy.furniture_ids, device=labels.device),
    ).float()[:, None]
    footprint_channel = footprint.float()
    if footprint_channel.ndim == 3:
        footprint_channel = footprint_channel[:, None]
    control = torch.cat(
        [rooms, wall, opening, furniture, footprint_channel],
        dim=1,
    )
    return F.adaptive_max_pool2d(control, (int(size), int(size)))


class CoarseSupportRefiner(nn.Module):
    """Recover polygon-level room support under walls and furniture."""

    def __init__(self, input_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        base = int(base_channels)
        self.enc128 = _ResidualBlock(input_channels, base)
        self.enc64 = _ResidualBlock(base, base * 2)
        self.enc32 = _ResidualBlock(base * 2, base * 4)
        self.dec64 = _ResidualBlock(base * 6, base * 2)
        self.dec128 = _ResidualBlock(base * 3, base)
        self.dec256 = _ResidualBlock(base, base)
        self.support = nn.Conv2d(base, 1, 1)

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
    ) -> torch.Tensor:
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
        return self.support(hidden256)


__all__ = ["CoarseSupportRefiner", "coarse_support_control"]

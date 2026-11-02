"""Layered CAD output that keeps semantic branches independent."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy


@dataclass
class LayeredFloorplan:
    """Room, wall, opening, and furniture outputs before raster composition."""

    room_labels: torch.Tensor
    wall: torch.Tensor
    opening: torch.Tensor
    furniture: torch.Tensor
    footprint: torch.Tensor
    taxonomy: FloorplanTaxonomy

    def __post_init__(self) -> None:
        self.footprint = self.footprint.bool()
        for name in ("room_labels", "wall", "opening", "furniture"):
            value = getattr(self, name)
            if value.ndim != 3 or value.shape != self.footprint.shape:
                raise ValueError(f"{name} must match footprint [B,H,W]")

    def visible_labels(self) -> torch.Tensor:
        """Compose layers for raster export without mutating their storage."""

        labels = self.room_labels.clone()
        labels = torch.where(self.furniture != 0, self.furniture, labels)
        wall_id = self.taxonomy.class_id("wall")
        labels = torch.where(
            self.wall.bool(),
            torch.full_like(labels, wall_id),
            labels,
        )
        labels = torch.where(self.opening != 0, self.opening, labels)
        return labels * self.footprint.long()

    def room_support(self) -> torch.Tensor:
        room_ids = torch.tensor(
            self.taxonomy.room_ids,
            device=self.room_labels.device,
            dtype=self.room_labels.dtype,
        )
        return torch.isin(self.room_labels, room_ids) & self.footprint


__all__ = ["LayeredFloorplan"]

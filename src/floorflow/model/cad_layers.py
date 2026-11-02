"""First-class CAD layer contract for FloorFlow predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.layered_floorplan import LayeredFloorplan


@dataclass(frozen=True)
class CadLayeredOutput:
    """Independent room, wall, opening, furniture, and support layers."""

    room_logits: torch.Tensor
    wall_logits: torch.Tensor
    opening_logits: torch.Tensor
    furniture_logits: torch.Tensor
    support_logits: torch.Tensor | None = None

    def resized(self, size: tuple[int, int]) -> CadLayeredOutput:
        def resize(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return F.interpolate(
                value,
                size=size,
                mode="bilinear",
                align_corners=False,
            )

        return CadLayeredOutput(
            room_logits=resize(self.room_logits),
            wall_logits=resize(self.wall_logits),
            opening_logits=resize(self.opening_logits),
            furniture_logits=resize(self.furniture_logits),
            support_logits=resize(self.support_logits),
        )

    def render(
        self,
        footprint: torch.Tensor,
        taxonomy: FloorplanTaxonomy,
        *,
        wall_threshold: float = 0.5,
        opening_threshold: float = 0.5,
        furniture_threshold: float = 0.5,
        condition_openings_on_walls: bool = True,
        condition_furniture_on_rooms: bool = True,
    ) -> LayeredFloorplan:
        room_ids = torch.tensor(
            taxonomy.room_ids,
            device=self.room_logits.device,
        )
        room_labels = room_ids[self.room_logits.argmax(dim=1)] * footprint.long()
        wall = self.wall_logits[:, 0].sigmoid() >= float(wall_threshold)
        opening_probability = self.opening_logits.softmax(dim=1)
        opening_ids = torch.tensor(
            taxonomy.opening_ids,
            device=self.opening_logits.device,
        )
        opening_class = opening_ids[opening_probability.argmax(dim=1)]
        opening_mask = (
            opening_probability.amax(dim=1) > float(opening_threshold)
        )
        if condition_openings_on_walls:
            wall_band = F.max_pool2d(
                wall.float()[:, None],
                5,
                stride=1,
                padding=2,
            )[:, 0] > 0.5
            opening_mask = opening_mask & wall_band
        opening = opening_class * opening_mask.long()
        furniture_probability = self.furniture_logits.softmax(dim=1)
        furniture_ids = torch.tensor(
            taxonomy.furniture_ids,
            device=self.furniture_logits.device,
        )
        furniture_class = furniture_ids[furniture_probability.argmax(dim=1)]
        furniture_mask = (
            furniture_probability.amax(dim=1) > float(furniture_threshold)
        )
        if condition_furniture_on_rooms:
            furniture_mask = furniture_mask & torch.isin(room_labels, room_ids)
        furniture = furniture_class * furniture_mask.long()
        return LayeredFloorplan(
            room_labels=room_labels,
            wall=wall.long(),
            opening=opening,
            furniture=furniture,
            footprint=footprint,
            taxonomy=taxonomy,
        )

    def visible_labels(
        self,
        footprint: torch.Tensor,
        taxonomy: FloorplanTaxonomy,
        **thresholds,
    ) -> torch.Tensor:
        return self.render(
            footprint,
            taxonomy,
            **thresholds,
        ).visible_labels()

    def scatter(
        self,
        taxonomy: FloorplanTaxonomy,
        *,
        channels: int | None = None,
    ) -> torch.Tensor:
        """Scatter layers into a compatibility class-logit tensor."""

        size = self.room_logits.shape[-2:]
        channels = int(channels or taxonomy.num_classes)
        output = self.room_logits.new_full(
            (self.room_logits.shape[0], channels, *size),
            -12.0,
        )
        room_ids = torch.tensor(
            taxonomy.room_ids,
            device=output.device,
        )
        output[:, room_ids] = self.room_logits
        output[:, taxonomy.class_id("wall")] = self.wall_logits[:, 0]
        opening_ids = torch.tensor(
            taxonomy.opening_ids,
            device=output.device,
        )
        output[:, opening_ids] = self.opening_logits
        furniture_ids = torch.tensor(
            taxonomy.furniture_ids,
            device=output.device,
        )
        output[:, furniture_ids] = self.furniture_logits
        if self.support_logits is not None:
            output[:, taxonomy.class_id("structure")] = self.support_logits[:, 0]
        return output


__all__ = ["CadLayeredOutput"]

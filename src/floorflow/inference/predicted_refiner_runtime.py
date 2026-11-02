"""Shared runtime for rendering a corrected predicted-control plan."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.joint_control_repair import JointControlRepair
from floorflow.representation.layered_floorplan import LayeredFloorplan


@torch.inference_mode()
def render_predicted_refiner(
    model: JointControlRepair,
    predicted: dict[str, torch.Tensor],
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    room_size: int,
    structure_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    room = F.interpolate(
        predicted["room_logits"],
        size=(room_size, room_size),
        mode="bilinear",
        align_corners=False,
    )
    structure = F.interpolate(
        predicted["structure"],
        size=(structure_size, structure_size),
        mode="bilinear",
        align_corners=False,
    )
    output = model(
        room,
        structure,
        footprint,
        torch.zeros_like(room[:, :1]),
        torch.zeros_like(structure[:, :1]),
        program=predicted["program"],
    )
    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=footprint.device,
    )
    room_labels = room_ids[output["room_logits_full"].argmax(dim=1)]
    room_labels = F.interpolate(
        room_labels[:, None].float(),
        size=footprint.shape[-2:],
        mode="nearest",
    )[:, 0].long()
    room_labels *= footprint.long()
    wall = F.interpolate(
        output["structure"]["wall_logits"].sigmoid(),
        size=footprint.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0] >= 0.5
    opening_type = F.interpolate(
        output["structure"]["opening_type_logits"],
        size=footprint.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)
    opening_ids = torch.tensor(
        taxonomy.opening_ids,
        device=footprint.device,
    )
    opening = opening_ids[opening_type] * (
        F.interpolate(
            output["structure"]["opening_logits"].sigmoid(),
            size=footprint.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0] >= 0.5
    )
    visible = LayeredFloorplan(
        room_labels=room_labels,
        wall=wall.long(),
        opening=opening,
        furniture=torch.zeros_like(room_labels),
        footprint=footprint,
        taxonomy=taxonomy,
    ).visible_labels()
    return visible, room_labels, output["room_logits_full"]


__all__ = ["render_predicted_refiner"]

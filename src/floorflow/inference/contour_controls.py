"""Contour-only stage-1 controls for the multi-stage CAD pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.flow.core import sample_floorplans
from floorflow.heads.room_mask_decoder import fused_room_mask_logits


@dataclass
class ContourControls:
    room_logits: torch.Tensor
    program: torch.Tensor
    structure: torch.Tensor
    furniture: torch.Tensor
    support: torch.Tensor
    auxiliary: dict


def resize_probability(
    value: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    return F.interpolate(
        value,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


def opening_probabilities(
    objects: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = objects["opening_logits"]
    background = objects["opening_background_logits"][:, None]
    probability = torch.cat([background, logits], dim=1).softmax(dim=1)
    foreground = 1.0 - probability[:, 0]
    class_probability = probability[:, 1:]
    class_probability = class_probability / class_probability.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    return foreground, class_probability


def furniture_probabilities(
    objects: dict[str, torch.Tensor],
) -> torch.Tensor:
    logits = objects["furniture_logits"]
    background = objects["furniture_background_logits"][:, None]
    probability = torch.cat([background, logits], dim=1).softmax(dim=1)
    class_probability = probability[:, 1:]
    return class_probability / class_probability.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)


def room_probabilities(
    room_logits: torch.Tensor,
) -> torch.Tensor:
    return room_logits.softmax(dim=1)


def build_structure_control(
    rooms: torch.Tensor,
    wall: torch.Tensor,
    opening: torch.Tensor,
    opening_types: torch.Tensor,
    footprint: torch.Tensor,
    *,
    size: int,
) -> torch.Tensor:
    control = torch.cat(
        [
            rooms,
            wall[:, None],
            opening[:, None],
            opening_types,
            footprint[:, None].float(),
        ],
        dim=1,
    )
    return resize_probability(control, (int(size), int(size)))


def build_furniture_control(
    rooms: torch.Tensor,
    furniture: torch.Tensor,
    footprint: torch.Tensor,
    *,
    size: int,
) -> torch.Tensor:
    control = torch.cat(
        [rooms, furniture, footprint[:, None].float()],
        dim=1,
    )
    return resize_probability(control, (int(size), int(size)))


def build_support_control(
    rooms: torch.Tensor,
    wall: torch.Tensor,
    opening: torch.Tensor,
    furniture: torch.Tensor,
    footprint: torch.Tensor,
    *,
    size: int,
) -> torch.Tensor:
    control = torch.cat(
        [
            rooms,
            wall[:, None],
            opening[:, None],
            furniture.sum(dim=1, keepdim=True).clamp(0.0, 1.0),
            footprint[:, None].float(),
        ],
        dim=1,
    )
    return resize_probability(control, (int(size), int(size)))


def controls_from_target_channels(
    state: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    structure_size: int = 128,
    furniture_size: int = 192,
    support_size: int = 192,
) -> ContourControls:
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    room_ids = torch.tensor(taxonomy.room_ids, device=state.device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=state.device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=state.device)
    room_scores = state[:, room_ids - 1].clamp_min(0.0)
    rooms = room_scores / room_scores.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    opening_types = state[:, opening_ids - 1].clamp_min(0.0)
    opening = opening_types.amax(dim=1)
    opening_types = opening_types / opening_types.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    furniture = state[:, furniture_ids - 1].clamp_min(0.0)
    furniture = furniture / furniture.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    wall = state[:, taxonomy.class_id("wall") - 1].clamp(0.0, 1.0)
    program = room_scores.amax(dim=(2, 3)) > 0.1
    missing = ~program.any(dim=1)
    if missing.any():
        fallback = room_scores.amax(dim=(2, 3)).argmax(dim=1)
        program = program.clone()
        program[missing, fallback[missing]] = True
    return ContourControls(
        room_logits=room_scores * 4.0 - 2.0,
        program=program,
        structure=build_structure_control(
            rooms,
            wall,
            opening,
            opening_types,
            footprint[:, 0],
            size=structure_size,
        ),
        furniture=build_furniture_control(
            rooms,
            furniture,
            footprint[:, 0],
            size=furniture_size,
        ),
        support=build_support_control(
            rooms,
            wall,
            opening,
            furniture,
            footprint[:, 0],
            size=support_size,
        ),
        auxiliary={},
    )


@torch.no_grad()
def predict_contour_controls(
    model,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    floor_ids: torch.Tensor | None = None,
    flow_steps: int = 12,
    noise_scale: float = 0.75,
    structure_size: int = 128,
    furniture_size: int = 192,
    support_size: int = 192,
    control_source: str = "aux",
    initial_state: torch.Tensor | None = None,
    use_room_instance_head: bool = True,
) -> ContourControls:
    if control_source not in {"aux", "sampled", "blend"}:
        raise ValueError("control_source must be aux, sampled, or blend")
    if footprint.ndim == 3:
        footprint_bchw = footprint[:, None]
    else:
        footprint_bchw = footprint
    sampled_state = False
    if initial_state is not None:
        if initial_state.shape[-2:] != footprint_bchw.shape[-2:]:
            raise ValueError("initial_state spatial shape does not match footprint")
        state = initial_state * footprint_bchw.float()
        sampled_state = True
    elif flow_steps > 0:
        if hasattr(model, "base"):
            raise ValueError("sparse-control wrappers require initial_state")
        state, _trajectory = sample_floorplans(
            model,
            footprint_bchw[:, 0],
            steps=int(flow_steps),
            noise_scale=float(noise_scale),
            floor_ids=floor_ids,
            objective_mode="velocity",
            semantic_decode_weight=0.0,
        )
        sampled_state = True
    else:
        state = torch.zeros(
            footprint_bchw.shape[0],
            model.target_channels,
            *footprint_bchw.shape[-2:],
            device=footprint_bchw.device,
            dtype=footprint_bchw.dtype,
        )
    forward_kwargs = {}
    if hasattr(model, "base"):
        forward_kwargs = {
            "control": state,
            "control_mask": torch.zeros_like(footprint_bchw, dtype=torch.bool),
        }
    _, auxiliary = model.forward_with_aux(
        state,
        torch.full(
            (footprint_bchw.shape[0],),
            1.0 if sampled_state else 0.0,
            device=footprint_bchw.device,
            dtype=footprint_bchw.dtype,
        ),
        footprint_bchw,
        floor_ids=floor_ids,
        **forward_kwargs,
    )
    room_logits = auxiliary.get("room_partition_logits")
    if room_logits is None:
        raise ValueError("stage-1 model does not expose room partition logits")
    room_mask_predictions = auxiliary.get("room_mask_predictions")
    if use_room_instance_head and room_mask_predictions is not None:
        room_logits = fused_room_mask_logits(
            room_mask_predictions,
            len(taxonomy.room_ids),
        )
    rooms = room_probabilities(room_logits)
    objects = auxiliary.get("structured_objects") or {}
    if "opening_logits" not in objects or "wall_logits" not in objects:
        raise ValueError("stage-1 model does not expose structural object heads")
    opening, opening_types = opening_probabilities(objects)
    furniture = furniture_probabilities(objects)
    wall = objects["wall_logits"].sigmoid()
    geometry = auxiliary.get("geometry") or {}
    geometry_wall = geometry.get("wall_centerline_logits")
    if geometry_wall is not None:
        wall = torch.maximum(
            wall,
            F.interpolate(
                geometry_wall.sigmoid(),
                size=wall.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0],
        )
    if sampled_state and control_source != "aux":
        room_ids = torch.tensor(taxonomy.room_ids, device=state.device)
        opening_ids = torch.tensor(taxonomy.opening_ids, device=state.device)
        furniture_ids = torch.tensor(taxonomy.furniture_ids, device=state.device)
        sampled_room = state[:, room_ids - 1]
        sampled_opening = state[:, opening_ids - 1].amax(dim=1)
        sampled_types = state[:, opening_ids - 1]
        sampled_types = sampled_types / sampled_types.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        sampled_furniture = state[:, furniture_ids - 1]
        sampled_furniture = sampled_furniture / sampled_furniture.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        sampled_wall = state[:, taxonomy.class_id("wall") - 1 : taxonomy.class_id("wall")]
        sampled_wall = sampled_wall[:, 0]
        if control_source == "sampled":
            room_logits = sampled_room * 4.0 - 2.0
            rooms = sampled_room / sampled_room.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0e-6)
            wall = sampled_wall
            opening = sampled_opening
            opening_types = sampled_types
            furniture = sampled_furniture
        else:
            room_logits = 0.5 * room_logits + 2.0 * sampled_room - 1.0
            rooms = rooms * 0.5 + 0.5 * sampled_room / sampled_room.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0e-6)
            wall = 0.5 * wall + 0.5 * sampled_wall
            opening = 0.5 * opening + 0.5 * sampled_opening
            opening_types = 0.5 * opening_types + 0.5 * sampled_types
            furniture = 0.5 * furniture + 0.5 * sampled_furniture
    program_predictions = auxiliary.get("room_program")
    if program_predictions is not None:
        program = program_predictions["presence_logits"].sigmoid() > 0.5
    else:
        program = F.adaptive_avg_pool2d(rooms, (8, 8)).amax(dim=(2, 3)) > 0.25
    missing = ~program.any(dim=1)
    if missing.any():
        fallback = rooms.amax(dim=(2, 3)).argmax(dim=1)
        program = program.clone()
        program[missing, fallback[missing]] = True
    return ContourControls(
        room_logits=room_logits,
        program=program,
        structure=build_structure_control(
            rooms,
            wall,
            opening,
            opening_types,
            footprint_bchw[:, 0],
            size=structure_size,
        ),
        furniture=build_furniture_control(
            rooms,
            furniture,
            footprint_bchw[:, 0],
            size=furniture_size,
        ),
        support=build_support_control(
            rooms,
            wall,
            opening,
            furniture,
            footprint_bchw[:, 0],
            size=support_size,
        ),
        auxiliary=auxiliary,
    )


__all__ = [
    "ContourControls",
    "controls_from_target_channels",
    "predict_contour_controls",
]

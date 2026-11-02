"""Mask-aware joint repair of missing coarse room and structure controls."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.heads.coarse_layout_refiner import CoarseLayoutRefiner
from floorflow.heads.coarse_structure_refiner import CoarseStructureRefiner
from floorflow.heads.structure_context import (
    STRUCTURE_CONTEXT_NAMES,
    build_structure_context,
)


def fuse_valid_regions(
    supplied: torch.Tensor,
    predicted: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    """Copy supplied values where valid and predicted values where missing."""

    if supplied.shape != predicted.shape:
        raise ValueError("supplied and predicted tensors must share a shape")
    if validity.ndim == 3:
        validity = validity[:, None]
    if validity.shape[-2:] != supplied.shape[-2:]:
        raise ValueError("validity mask must match control resolution")
    return torch.where(validity.bool(), supplied, predicted)


class JointControlRepair(nn.Module):
    """Repair room logits and CAD structure with exact valid-region fusion."""

    def __init__(
        self,
        num_room_classes: int,
        num_opening_classes: int,
        *,
        room_base_channels: int = 24,
        structure_base_channels: int = 32,
        structure_contract: str = "legacy",
        predict_centerline: bool = False,
        host_conditioning_v2: bool = False,
        structure_global_route_attention: bool = False,
        structure_route_grid: int = 16,
        structure_route_router_count: int = 16,
        structure_route_top_k: int = 8,
        structure_route_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_room_classes = int(num_room_classes)
        self.num_opening_classes = int(num_opening_classes)
        if structure_contract not in {"legacy", "typed_v2"}:
            raise ValueError(
                "structure_contract must be legacy or typed_v2"
            )
        self.structure_contract = structure_contract
        self.typed_structure = structure_contract == "typed_v2"
        self.room_refiner = CoarseLayoutRefiner(
            self.num_room_classes,
            base_channels=room_base_channels,
            residual=True,
            masked_input=True,
            zero_masked_input=False,
        )
        structure_channels = (
            self.num_room_classes + self.num_opening_classes + 3
        )
        self.structure_refiner = CoarseStructureRefiner(
            structure_channels,
            self.num_opening_classes,
            base_channels=structure_base_channels,
            masked_input=True,
            zero_masked_input=False,
            extra_input_channels=(
                len(STRUCTURE_CONTEXT_NAMES) if self.typed_structure else 0
            ),
            direct_control_output=self.typed_structure,
            host_conditioned_openings=self.typed_structure,
            host_conditioning_v2=bool(host_conditioning_v2),
            predict_junctions=self.typed_structure,
            predict_room_boundary=self.typed_structure,
            predict_centerline=bool(predict_centerline),
            global_route_attention=bool(
                structure_global_route_attention
            ),
            route_grid=int(structure_route_grid),
            route_router_count=int(structure_route_router_count),
            route_top_k=int(structure_route_top_k),
            route_heads=int(structure_route_heads),
        )

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask[:, None]
        return F.interpolate(mask.float(), size=size, mode="nearest")

    def _structure_control(
        self,
        room_probability: torch.Tensor,
        structure: dict[str, torch.Tensor],
        footprint: torch.Tensor,
    ) -> torch.Tensor:
        opening_probability = structure["opening_logits"].sigmoid()
        opening_type = structure["opening_type_logits"].softmax(dim=1)
        return torch.cat(
            [
                room_probability,
                structure["wall_logits"].sigmoid(),
                opening_probability,
                opening_type,
                footprint,
            ],
            dim=1,
        )

    def forward(
        self,
        room_logits: torch.Tensor,
        structure_control: torch.Tensor,
        footprint: torch.Tensor,
        room_validity: torch.Tensor,
        structure_validity: torch.Tensor,
        program: torch.Tensor | None = None,
        repair_room: bool = True,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        room_validity = self._resize_mask(
            room_validity,
            room_logits.shape[-2:],
        )
        structure_validity = self._resize_mask(
            structure_validity,
            structure_control.shape[-2:],
        )
        if repair_room:
            conditioned_room = (
                torch.where(
                    program.bool()[:, :, None, None],
                    room_logits,
                    torch.full_like(room_logits, -10.0),
                )
                if program is not None
                else room_logits
            )
            repaired_room_full = self.room_refiner(
                conditioned_room,
                footprint,
                room_validity,
            )
            repaired_room = F.interpolate(
                repaired_room_full,
                size=room_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            room_control = fuse_valid_regions(
                room_logits,
                repaired_room,
                room_validity,
            )
        else:
            repaired_room_full = room_logits
            room_control = room_logits

        room_probability_full = repaired_room_full.softmax(dim=1)
        if self.typed_structure:
            room_probability = F.interpolate(
                room_probability_full,
                size=structure_control.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            room_probability = room_probability / room_probability.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0e-8)
            context, context_diagnostics = build_structure_context(
                room_probability,
                structure_control,
                self.num_room_classes,
                self.num_opening_classes,
            )
            structure_input = torch.cat(
                [structure_control, context],
                dim=1,
            )
        else:
            room_probability = F.adaptive_max_pool2d(
                room_probability_full,
                structure_control.shape[-2:],
            )
            structure_input = structure_control.clone()
            structure_input[:, : self.num_room_classes] = room_probability
            context_diagnostics = {}
        structure_input[:, : self.num_room_classes] = room_probability
        structure = self.structure_refiner(
            structure_input,
            output_size=structure_control.shape[-2:],
            validity_mask=structure_validity,
        )
        structure_footprint = F.interpolate(
            footprint.float(),
            size=structure_control.shape[-2:],
            mode="nearest",
        )
        predicted_structure = self._structure_control(
            room_probability,
            structure,
            structure_footprint,
        )
        structure_output = fuse_valid_regions(
            structure_control,
            predicted_structure,
            structure_validity,
        )
        return {
            "room_logits": room_control,
            "room_logits_full": repaired_room_full,
            "structure": structure,
            "structure_control": structure_output,
            "predicted_structure_control": predicted_structure,
            "structure_context_diagnostics": context_diagnostics,
        }


__all__ = ["JointControlRepair", "fuse_valid_regions"]

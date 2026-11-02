"""Reusable predicted-control stage heads for the CAD action graph."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.joint_control_loading import load_joint_control_repair
from floorflow.heads.raster_control_bank import (
    blend_room_instance_probabilities,
    load_raster_instance_control_model,
)
from floorflow.inference.opening_heatmap_decoder import (
    load_opening_heatmap_decoder,
    predict_opening_detection_maps,
)
from floorflow.inference.structure_evidence import (
    load_geometry_wall_model,
    wall_geometry_probabilities,
)
from floorflow.representation.program_conditioning import apply_program_constraint


def validate_stage_controls(
    controls: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
) -> None:
    """Fail before inference when cached stage controls violate the taxonomy."""

    room_count = len(taxonomy.room_ids)
    expected_structure_channels = (
        room_count + len(taxonomy.opening_ids) + 3
    )
    if "structure" not in controls or "program" not in controls:
        raise ValueError("stage runtime requires structure and program controls")
    if controls["structure"].ndim != 4:
        raise ValueError("structure control must be [B,C,H,W]")
    if controls["structure"].shape[1] != expected_structure_channels:
        raise ValueError(
            "structure control channel count does not match taxonomy: "
            f"{controls['structure'].shape[1]} != "
            f"{expected_structure_channels}"
        )
    if controls["program"].ndim != 2:
        raise ValueError("program control must be [B,room_classes]")
    if controls["program"].shape[1] != room_count:
        raise ValueError("program width does not match the room taxonomy")


def fuse_learned_opening_consumer(
    structure: torch.Tensor,
    structure_opening_logits: torch.Tensor,
    structure_opening_type_logits: torch.Tensor,
    learned_opening_probability: torch.Tensor | None,
    taxonomy: FloorplanTaxonomy,
    *,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Replace only the opening channels, preserving every other CAD layer."""

    output = structure.clone()
    if not enabled:
        return output, None
    if learned_opening_probability is None:
        raise ValueError("enabled opening consumer requires learned probabilities")
    room_count = len(taxonomy.room_ids)
    family = F.interpolate(
        learned_opening_probability.amax(dim=1)[:, None].float(),
        size=output.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    output[:, room_count + 1] = torch.maximum(
        structure_opening_logits.sigmoid()[:, 0],
        family,
    )
    opening_channels = room_count + 2, room_count + 2 + len(taxonomy.opening_ids)
    output[:, opening_channels[0] : opening_channels[1]] = (
        structure_opening_type_logits.softmax(dim=1)
    )
    return output, family


def opening_control_maps(
    maps: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Bind detector maps without collapsing foreground into family scores."""

    if maps is None:
        return {}
    required = ("family_probability", "foreground", "family_logits")
    missing = [name for name in required if name not in maps]
    if missing:
        raise ValueError(f"opening detector maps missing: {missing}")
    return {
        "opening_family_probability": maps["family_probability"],
        "opening_foreground_probability": maps["foreground"],
        "opening_family_logits": maps["family_logits"],
    }


def fuse_junction_evidence(
    geometry_junction: torch.Tensor,
    structure_junction_logits: torch.Tensor | None,
) -> torch.Tensor:
    """Fuse geometry and structure junction evidence as probabilities."""

    if geometry_junction.ndim != 3:
        raise ValueError("geometry junction must be [B,H,W]")
    if structure_junction_logits is None:
        return geometry_junction
    if structure_junction_logits.ndim != 4 or structure_junction_logits.shape[1] != 1:
        raise ValueError("structure junction logits must be [B,1,H,W]")
    structure_junction = F.interpolate(
        structure_junction_logits.float().sigmoid(),
        size=geometry_junction.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return torch.maximum(geometry_junction.float(), structure_junction)


class CadStageRuntime:
    """Run frozen room, wall, junction, and opening stage predictors."""

    def __init__(
        self,
        taxonomy: FloorplanTaxonomy,
        device: torch.device,
        *,
        room_checkpoint: str | Path,
        structure_checkpoint: str | Path,
        centerline_checkpoint: str | Path,
        opening_checkpoint: str | Path | None,
        room_blend: float = 0.5,
        room_size: int = 224,
        structure_size: int = 128,
        room_program_bias: float = 0.0,
        use_junction_stage: bool = False,
        repair_room: bool = False,
        use_opening_consumer: bool = True,
    ) -> None:
        self.taxonomy = taxonomy
        self.device = device
        self.room_blend = float(room_blend)
        self.room_size = int(room_size)
        self.structure_size = int(structure_size)
        self.room_program_bias = float(room_program_bias)
        self.use_junction_stage = bool(use_junction_stage)
        self.repair_room = bool(repair_room)
        self.use_opening_consumer = bool(use_opening_consumer)
        self.room = load_raster_instance_control_model(
            room_checkpoint,
            taxonomy,
            device,
        ).requires_grad_(False).eval()
        self.structure, _structure_config = load_joint_control_repair(
            structure_checkpoint,
            taxonomy,
            device,
        )
        self.structure.requires_grad_(False)
        self.centerline = load_geometry_wall_model(
            centerline_checkpoint,
            device,
            taxonomy,
        )
        if self.use_opening_consumer:
            if opening_checkpoint is None:
                raise ValueError(
                    "opening consumer requires an opening checkpoint"
                )
            self.opening = load_opening_heatmap_decoder(
                opening_checkpoint,
                taxonomy,
                device,
            )
        else:
            self.opening = None

    @torch.inference_mode()
    def __call__(
        self,
        controls: dict[str, torch.Tensor],
        footprint: torch.Tensor,
        image: torch.Tensor,
        floor_ids: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        validate_stage_controls(controls, self.taxonomy)
        room_count = len(self.taxonomy.room_ids)
        room_output = self.room(footprint, control_image=image)
        room_probability = blend_room_instance_probabilities(
            room_output,
            self.room_blend,
        )
        room_probability = room_probability.clamp_min(1.0e-8)
        program = controls.get("program")
        room_program_applied = False
        if (
            program is not None
            and self.room_program_bias != 0.0
            and program.shape[1] == room_probability.shape[1]
            and bool(program.any())
        ):
            room_probability = apply_program_constraint(
                room_probability.log(),
                program,
                absent_penalty=self.room_program_bias,
            ).softmax(dim=1)
            room_program_applied = True
        room_logits = F.interpolate(
            room_probability.log(),
            size=(self.room_size, self.room_size),
            mode="bilinear",
            align_corners=False,
        )
        structure_input = F.interpolate(
            controls["structure"].float(),
            size=(self.structure_size, self.structure_size),
            mode="bilinear",
            align_corners=False,
        )
        room_probability_structure = F.interpolate(
            room_probability,
            size=structure_input.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        structure_input[:, : len(self.taxonomy.room_ids)] = room_probability_structure
        structure_output = self.structure(
            room_logits,
            structure_input,
            footprint,
            torch.zeros_like(room_logits[:, :1]),
            torch.zeros(
                footprint.shape[0],
                1,
                self.structure_size,
                self.structure_size,
                device=footprint.device,
            ),
            program=controls["program"],
            repair_room=self.repair_room,
        )["structure"]
        centerline, junction = wall_geometry_probabilities(
            self.centerline,
            torch.zeros_like(footprint, dtype=torch.long),
            footprint,
            floor_ids,
        )
        centerline = F.interpolate(
            centerline[:, None].float(),
            size=structure_input.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        junction = F.interpolate(
            junction[:, None].float(),
            size=structure_input.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        wall_probability = 0.5 * (
            structure_output["wall_logits"].sigmoid()[:, 0] + centerline
        )
        junction = fuse_junction_evidence(
            junction,
            structure_output.get("junction_logits"),
        )
        wall_probability = torch.maximum(wall_probability, junction)
        opening_maps = (
            predict_opening_detection_maps(
                self.opening,
                footprint,
                image,
                self.taxonomy,
            )
            if self.opening is not None
            else None
        )
        opening_probability = (
            opening_maps["family_probability"]
            if opening_maps is not None
            else None
        )
        structure_input[:, room_count] = wall_probability
        structure_input, opening_family = fuse_learned_opening_consumer(
            structure_input,
            structure_output["opening_logits"],
            structure_output["opening_type_logits"],
            opening_probability,
            self.taxonomy,
            enabled=self.use_opening_consumer,
        )
        result = dict(controls)
        result["room_logits"] = room_logits
        result["structure"] = structure_input
        result.update(opening_control_maps(opening_maps))
        # The geometry branch is already executed above. Retaining its junction
        # map is the typed topological consumer; the historical flag cannot
        # discard a computed signal without changing compute.
        result["junction_probability"] = junction
        diagnostics = {
            "room_confidence": room_probability.amax(dim=1).mean(),
            "wall_probability": wall_probability.mean(),
            "centerline_probability": centerline.mean(),
            "junction_probability": junction.mean(),
            "room_program_applied": torch.tensor(
                float(room_program_applied),
                device=footprint.device,
            ),
            "room_repair": torch.tensor(
                float(self.repair_room),
                device=footprint.device,
            ),
            "opening_probability": (
                opening_family.mean()
                if opening_family is not None
                else footprint.new_tensor(0.0)
            ),
            "opening_consumer_enabled": torch.tensor(
                float(self.use_opening_consumer),
                device=footprint.device,
            ),
        }
        return result, diagnostics


__all__ = [
    "CadStageRuntime",
    "fuse_junction_evidence",
    "fuse_learned_opening_consumer",
    "opening_control_maps",
    "validate_stage_controls",
]

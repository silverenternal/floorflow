"""Load and apply reusable wall/opening structure-evidence models."""

from __future__ import annotations

from pathlib import Path

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.structured_objects import (
    StructuredObjectHeads,
    object_group_predictions,
)
from floorflow.inference.layered_structure_readout import (
    build_layered_structure_plan,
)
from floorflow.inference.opening_evidence_fusion import fuse_opening_evidence
from floorflow.inference.wall_centerline_fusion import fuse_wall_centerline
from floorflow.training.checkpoint import load_checkpoint_model


def load_geometry_wall_model(
    checkpoint: str | Path,
    device: torch.device,
    taxonomy: FloorplanTaxonomy,
):
    model, _config, _taxonomy, _state = load_checkpoint_model(
        checkpoint,
        device,
        taxonomy=taxonomy,
    )
    if not hasattr(model, "geometry_heads"):
        raise ValueError("checkpoint has no geometry heads")
    return model.requires_grad_(False).eval()


def wall_geometry_probabilities(
    model,
    target: torch.Tensor,
    footprint: torch.Tensor,
    floor_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero_target = torch.zeros(
        target.shape[0],
        model.target_channels,
        *target.shape[-2:],
        device=target.device,
    )
    features = model._encode_features(
        zero_target,
        torch.zeros(target.shape[0], device=target.device),
        footprint,
        floor_ids=floor_ids,
    )
    outputs = model.geometry_heads(features)
    return (
        outputs["wall_centerline_logits"][:, 0].sigmoid(),
        outputs["junction_logits"][:, 0].sigmoid(),
    )


def wall_centerline_probability(
    model,
    target: torch.Tensor,
    footprint: torch.Tensor,
    floor_ids: torch.Tensor,
) -> torch.Tensor:
    return wall_geometry_probabilities(
        model,
        target,
        footprint,
        floor_ids,
    )[0]


def load_structured_opening_model(
    checkpoint: str | Path,
    device: torch.device,
    taxonomy: FloorplanTaxonomy,
):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone_path = payload.get("backbone_checkpoint")
    if backbone_path is None:
        raise ValueError("structured-object checkpoint has no backbone path")
    backbone, _config, _taxonomy, _state = load_checkpoint_model(
        backbone_path,
        device,
        taxonomy=taxonomy,
    )
    head = StructuredObjectHeads(backbone.output.in_channels, taxonomy).to(device)
    head.load_state_dict(payload["head"])
    return backbone.requires_grad_(False).eval(), head.eval()


def structured_opening_probability(
    backbone,
    head: StructuredObjectHeads,
    target: torch.Tensor,
    footprint: torch.Tensor,
    floor_ids: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float,
) -> torch.Tensor:
    zero_target = torch.zeros(
        target.shape[0],
        backbone.target_channels,
        *target.shape[-2:],
        device=target.device,
    )
    features = backbone._encode_features(
        zero_target,
        torch.zeros(target.shape[0], device=target.device),
        footprint,
        floor_ids=floor_ids,
    )
    outputs = head(features)
    predictions = object_group_predictions(
        outputs,
        taxonomy,
        opening_threshold=threshold,
        furniture_threshold=1.1,
        wall_threshold=1.1,
        score_scale="probability",
    )
    return predictions["opening"]


def apply_structure_evidence(
    labels: torch.Tensor,
    room_labels: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    floor_ids: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    geometry_model=None,
    object_backbone=None,
    object_head=None,
    wall_threshold: float = 0.5,
    wall_dilation: int = 1,
    opening_protect_radius: int = 3,
    wall_mode: str = "consensus",
    opening_threshold: float = 0.5,
):
    prediction_plans = None
    if geometry_model is not None:
        centerline = wall_centerline_probability(
            geometry_model,
            target,
            footprint,
            floor_ids,
        )
        if wall_mode == "separate":
            wall_labels = torch.zeros_like(labels)
            wall_labels[centerline >= wall_threshold] = taxonomy.class_id("wall")
            prediction_plans = [
                build_layered_structure_plan(
                    room_labels[index].cpu().numpy(),
                    wall_labels[index].cpu().numpy(),
                    footprint[index].cpu().numpy(),
                    taxonomy,
                )
                for index in range(room_labels.shape[0])
            ]
        else:
            labels, _ = fuse_wall_centerline(
                labels,
                room_labels,
                footprint,
                centerline,
                taxonomy,
                threshold=wall_threshold,
                dilation=wall_dilation,
                opening_protect_radius=opening_protect_radius,
                mode=wall_mode,
            )
    if object_backbone is not None and object_head is not None:
        opening = structured_opening_probability(
            object_backbone,
            object_head,
            target,
            footprint,
            floor_ids,
            taxonomy,
            threshold=opening_threshold,
        )
        labels = fuse_opening_evidence(labels, footprint, opening, taxonomy)
    return labels, prediction_plans


__all__ = [
    "apply_structure_evidence",
    "load_geometry_wall_model",
    "load_structured_opening_model",
    "structured_opening_probability",
    "wall_centerline_probability",
    "wall_geometry_probabilities",
]

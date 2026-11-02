"""Reusable target-free contour-to-program-to-layout inference adapters."""

from __future__ import annotations

from pathlib import Path

import torch

from floorflow.heads.contour_program_layout import ContourProgramLayoutPredictor
from floorflow.heads.room_program import RoomProgramHead
from floorflow.heads.room_program_set import (
    RoomProgramSetHead,
    decode_room_program,
    load_room_program_set_state,
)
from floorflow.training.checkpoint import load_checkpoint_model


@torch.inference_mode()
def predict_room_program(backbone, head, footprint, floor_ids):
    """Predict room presence from footprint and floor metadata only."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    zero = footprint.new_zeros(
        footprint.shape[0],
        backbone.target_channels,
        *footprint.shape[-2:],
    )
    features = backbone._encode_features(
        zero,
        footprint.new_zeros(footprint.shape[0]),
        footprint,
        floor_ids=floor_ids,
    )
    output = head(features)
    if isinstance(head, RoomProgramSetHead):
        return decode_room_program(output) > 0.5
    return output["presence_logits"].sigmoid() > 0.5


def load_room_program_predictor(path, taxonomy, device):
    """Load the frozen contour-only room-program classifier."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    backbone, _config, _taxonomy, _state = load_checkpoint_model(
        Path(payload["backbone_checkpoint"]),
        device,
        taxonomy=taxonomy,
    )
    config = payload.get("config", {})
    head_type = payload.get(
        "head_type",
        "set" if "queries" in config else "bce",
    )
    if head_type == "set":
        head = RoomProgramSetHead(
            backbone.output.in_channels,
            len(taxonomy.room_ids),
            hidden_dim=int(config.get("hidden_dim", 96)),
            tokens=int(config.get("tokens", 8)),
            layers=int(config.get("layers", 1)),
            heads=int(config.get("heads", 4)),
            num_queries=int(config.get("queries", 32)),
        ).to(device)
        load_room_program_set_state(head, payload["head"])
    else:
        head = RoomProgramHead(
            backbone.output.in_channels,
            len(taxonomy.room_ids),
        ).to(device)
        head.load_state_dict(payload["head"])
    return backbone.eval().requires_grad_(False), head.eval().requires_grad_(False)


def load_contour_layout_predictor(path, num_room_classes, device):
    """Load a contour-and-program coarse room-layout predictor."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = ContourProgramLayoutPredictor(
        int(num_room_classes),
        base_channels=int(payload.get("config", {}).get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=False)
    return model.eval().requires_grad_(False)


__all__ = [
    "load_contour_layout_predictor",
    "load_room_program_predictor",
    "predict_room_program",
]

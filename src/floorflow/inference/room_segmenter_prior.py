"""Load and apply an independent contour-to-room segmentation prior."""

from __future__ import annotations

from pathlib import Path

import torch

from floorflow.heads.room_segmenter import ContourRoomSegmenter


def load_room_segmenter_prior(
    checkpoint_path: str | Path,
    taxonomy,
    device: torch.device,
) -> tuple[ContourRoomSegmenter, dict[str, object]]:
    """Restore a train-fitted segmenter while rejecting taxonomy drift."""

    payload = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_taxonomy = payload.get("taxonomy")
    checkpoint_names = (
        checkpoint_taxonomy.get("class_names")
        if isinstance(checkpoint_taxonomy, dict)
        else None
    )
    if checkpoint_names is not None and tuple(checkpoint_names) != tuple(
        taxonomy.class_names
    ):
        raise ValueError("room segmenter taxonomy does not match evaluator taxonomy")
    config = dict(payload.get("config", {}))
    decoder_channels = int(config.get("decoder_channels", 96))
    model = ContourRoomSegmenter(
        len(taxonomy.room_ids),
        pretrained=False,
        decoder_channels=decoder_channels,
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, config


@torch.inference_mode()
def predict_room_segmenter_prior(
    model: ContourRoomSegmenter,
    footprint: torch.Tensor,
) -> torch.Tensor:
    """Return room-class logits for a binary footprint batch."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    return model(footprint.to(next(model.parameters()).device))


__all__ = [
    "load_room_segmenter_prior",
    "predict_room_segmenter_prior",
]

"""Attach Mask2Former-style room instance heads to existing CFM checkpoints."""

from __future__ import annotations

from torch import nn

from floorflow.heads.room_mask_decoder import RoomMaskDecoder


def attach_room_mask_head(
    model,
    taxonomy,
    *,
    queries: int = 32,
    hidden_dim: int = 128,
    resolution: int = 64,
):
    """Add a zero-fusion room query head while preserving the original model."""

    if not isinstance(model.room_mask_decoder, nn.Identity):
        return model
    model.room_mask_decoder = RoomMaskDecoder(
        model.output.in_channels + len(taxonomy.room_ids),
        len(taxonomy.room_ids),
        hidden_dim=int(hidden_dim),
        num_queries=int(queries),
        feature_scales=(1, 2, 4),
    )
    model.use_room_mask_head = True
    model.room_mask_use_partition = True
    model.room_mask_resolution = int(resolution)
    return model


__all__ = ["attach_room_mask_head"]

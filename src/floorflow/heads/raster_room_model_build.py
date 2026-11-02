"""Factory for raster-conditioned dense room models."""

from __future__ import annotations

from pathlib import Path

from torch import nn


def build_raster_room_model(
    architecture: str,
    num_room_classes: int,
    *,
    decoder_channels: int = 96,
    backbone_weights: str | Path | None = None,
) -> nn.Module:
    from floorflow.heads.convnext_room_control import ConvNeXtRoomControl
    from floorflow.heads.raster_control_bank import DeepLabRasterRoomControl
    from floorflow.heads.room_segmenter import ContourRoomSegmenter

    if architecture == "resnet18_fpn":
        return ContourRoomSegmenter(
            num_room_classes,
            pretrained=False,
            decoder_channels=int(decoder_channels),
        )
    if architecture == "deeplabv3_resnet50":
        return DeepLabRasterRoomControl(
            num_room_classes,
            pretrained=backbone_weights is None,
            backbone_weights=backbone_weights,
        )
    if architecture == "convnext_tiny_fpn":
        return ConvNeXtRoomControl(
            num_room_classes,
            pretrained=backbone_weights is None,
            decoder_channels=decoder_channels,
            backbone_weights=backbone_weights,
        )
    raise ValueError(f"unknown raster room architecture: {architecture}")


__all__ = ["build_raster_room_model"]

"""Tests for raster instance-query training controls."""

from collections import Counter

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.raster_control_bank import RasterInstanceControl
from floorflow.training.raster_instance_control import (
    configure_room_class_recovery,
)


def test_class_recovery_freezes_geometry_and_opens_class_queries() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Bath": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=8,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        adaptive_query_refinement=True,
        mask_pooled_class_fusion=True,
        room_program_head=True,
        program_token_conditioning=True,
    )
    names = configure_room_class_recovery(model)
    assert names
    assert all(
        name.startswith(
            (
                "mask_decoder.class_fusion.",
                "mask_decoder.class_head.",
                "mask_decoder.decoder_norm.",
                "mask_decoder.adaptive_query_refiner.",
                "mask_pooled_class_fusion.",
                "room_program_head.",
            )
        )
        for name in names
    )
    assert not model.mask_decoder.mask_head[0].weight.requires_grad
    assert model.mask_decoder.class_head.weight.requires_grad

from types import SimpleNamespace

import torch

from floorflow.training.optimizer import (
    freeze_except_hierarchy_heads,
    make_optimizer,
)
from floorflow.model.backbone import FloorplanFlowModel


def test_hierarchy_and_polygon_parameters_use_head_learning_rate() -> None:
    model = FloorplanFlowModel(
        target_channels=4,
        base_channels=8,
        levels=2,
        semantic_channel_attention=False,
        slot_mixing=False,
        dynamic_output_head=False,
        topology_aux=False,
        floor_conditioning=False,
        semantic_endpoint_head=False,
        room_partition_head=True,
        num_room_classes=3,
        hierarchical_rooms=True,
        hierarchical_hidden=16,
        hierarchical_tokens=2,
        hierarchical_coarse_resolution=8,
        polygon_loop_head=True,
        polygon_queries=6,
        polygon_points=12,
        polygon_hidden=24,
        room_mask_head=True,
        room_mask_queries=6,
        room_mask_resolution=8,
    )
    args = SimpleNamespace(
        fused_adamw=False,
        lr=1.0e-5,
        head_lr=1.0e-3,
        weight_decay=0.0,
    )
    optimizer = make_optimizer(model, args)
    by_parameter = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            by_parameter[id(parameter)] = group["lr"]
    assert by_parameter[id(model.hierarchical_rooms.program.presence.weight)] == 1.0e-3
    assert by_parameter[id(model.polygon_loop.corner_head.weight)] == 1.0e-3
    assert by_parameter[id(model.room_mask_decoder.class_head.weight)] == 1.0e-3
    assert by_parameter[id(model.condition_encoder.stem.weight)] == 1.0e-5


def test_hierarchy_only_mode_keeps_mask_decoder_trainable() -> None:
    model = FloorplanFlowModel(
        target_channels=4,
        base_channels=8,
        levels=2,
        semantic_channel_attention=False,
        slot_mixing=False,
        dynamic_output_head=False,
        topology_aux=False,
        floor_conditioning=False,
        semantic_endpoint_head=False,
        room_partition_head=True,
        num_room_classes=3,
        hierarchical_rooms=True,
        hierarchical_hidden=16,
        hierarchical_tokens=2,
        hierarchical_coarse_resolution=8,
        room_mask_head=True,
        room_mask_queries=6,
        room_mask_resolution=8,
    )
    freeze_except_hierarchy_heads(model)
    assert model.room_mask_decoder.class_head.weight.requires_grad
    assert model.hierarchical_rooms.program.presence.weight.requires_grad
    assert not model.condition_encoder.stem.weight.requires_grad
    assert not model.room_partition.net[0].weight.requires_grad

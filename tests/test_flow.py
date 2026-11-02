from __future__ import annotations

import torch

from floorflow.flow.core import decode_floorplans, flow_matching_loss, sample_floorplans
from floorflow.geometry.core import compose_known_exterior
from floorflow.model.backbone import DynamicMultiScaleOutputHead, FloorplanFlowModel
from floorflow.representation.slots import gated_cross_attention, soft_slot_decompose
from floorflow.inference.structured_decode import decode_structured_floorplan
from floorflow.data.taxonomy import build_taxonomy
from collections import Counter


def test_flow_shapes_and_masking() -> None:
    model = FloorplanFlowModel(target_channels=12, base_channels=8, levels=2, condition_dropout=0.0)
    target = torch.zeros(2, 32, 32, dtype=torch.long)
    target[:, 8:24, 8:24] = 4
    target[:, 8:11, 8:24] = 1
    footprint = torch.zeros(2, 32, 32)
    footprint[:, 6:26, 6:26] = 1.0
    loss, parts = flow_matching_loss(model, target, footprint)
    assert torch.isfinite(loss)
    assert {"loss", "velocity_loss", "category_loss"}.issubset(parts)
    sampled, trajectory = sample_floorplans(model, footprint, steps=2, heun=True)
    assert sampled.shape == (2, 12, 32, 32)
    assert trajectory.ndim == 5
    assert torch.count_nonzero(sampled * (1.0 - footprint[:, None])) == 0
    labels = decode_floorplans(sampled)
    assert labels.shape == target.shape
    smoothed = decode_floorplans(sampled, smooth_kernel=3)
    assert smoothed.shape == labels.shape
    composed = compose_known_exterior(labels, footprint)
    assert torch.count_nonzero(composed * (footprint == 0)) == 0
    labels[0, 7, 10] = 11
    preserved = compose_known_exterior(labels, footprint, opening_ids=(11,))
    assert int(preserved[0, 7, 10]) == 11


def test_semantic_channel_cross_attention_is_active() -> None:
    model = FloorplanFlowModel(
        target_channels=12,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        semantic_channel_attention=True,
    )
    assert not isinstance(model.semantic_channel_attention, torch.nn.Identity)
    assert model.semantic_channel_attention.channel_prior.shape == (12, 12)
    footprint = torch.ones(2, 32, 32)
    x_t = torch.randn(2, 12, 32, 32)
    t = torch.rand(2)
    velocity = model(x_t, t, footprint)
    assert velocity.shape == x_t.shape


def test_multiflow_slot_decomposition_and_gated_attention() -> None:
    spatial = torch.randn(2, 64, 16)
    condition = torch.randn(2, 64, 16)
    spatial_slots = soft_slot_decompose(spatial, num_slots=8)
    condition_slots = soft_slot_decompose(condition, num_slots=8)
    path = gated_cross_attention(spatial_slots, condition_slots)
    assert spatial_slots.slots.shape == (2, 8, 16)
    assert path.attended.shape == spatial_slots.slots.shape
    assert 0.0 <= float(path.gate) <= 1.0


def test_balanced_object_loss_is_finite() -> None:
    model = FloorplanFlowModel(
        target_channels=19,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        semantic_channel_attention=True,
        slot_mixing=True,
    )
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 8:24, 8:24] = 4
    target[:, 14:16, 14:16] = 11
    footprint = torch.zeros(1, 32, 32)
    footprint[:, 6:26, 6:26] = 1.0
    loss, parts = flow_matching_loss(model, target, footprint, balanced_object_loss=True)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_dynamic_output_head_learns_scale_weights() -> None:
    head = DynamicMultiScaleOutputHead(8, 19, (3, 5, 7))
    features = torch.randn(2, 8, 32, 32)
    output = head(features)
    assert output.shape == (2, 19, 32, 32)
    logits = head.router(features).reshape(2, 3, 19, 32, 32)
    weights = torch.softmax(logits, dim=1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 19, 32, 32))


def test_dynamic_head_starts_as_fixed_head_identity() -> None:
    fixed = FloorplanFlowModel(
        target_channels=19,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        dynamic_output_head=False,
    )
    dynamic = FloorplanFlowModel(
        target_channels=19,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        dynamic_output_head=True,
    )
    dynamic.load_state_dict(fixed.state_dict(), strict=False)
    x_t = torch.randn(1, 19, 32, 32)
    footprint = torch.ones(1, 32, 32)
    t = torch.tensor([0.5])
    with torch.no_grad():
        assert torch.allclose(fixed(x_t, t, footprint), dynamic(x_t, t, footprint), atol=1.0e-6)


def test_simplex_objective_and_sampling() -> None:
    model = FloorplanFlowModel(
        target_channels=19,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        topology_aux=False,
    )
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 8:24, 8:24] = 4
    footprint = torch.zeros(1, 32, 32)
    footprint[:, 6:26, 6:26] = 1.0
    loss, parts = flow_matching_loss(model, target, footprint, objective_mode="simplex")
    assert torch.isfinite(loss)
    assert float(parts["objective_mode_simplex"]) == 1.0
    sampled, _ = sample_floorplans(model, footprint, steps=2, objective_mode="simplex")
    assert sampled.shape == (1, 19, 32, 32)


def test_classifier_free_guidance_sampling_shape() -> None:
    model = FloorplanFlowModel(
        target_channels=19,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        topology_aux=False,
        floor_conditioning=True,
    )
    footprint = torch.ones(1, 32, 32)
    sampled, _ = sample_floorplans(
        model,
        footprint,
        steps=2,
        guidance_scale=1.5,
        floor_ids=torch.tensor([1]),
    )
    assert sampled.shape == (1, 19, 32, 32)


def test_semantic_endpoint_fast_path_skips_flow_integration() -> None:
    model = FloorplanFlowModel(
        target_channels=12,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        dynamic_output_head=False,
        semantic_endpoint_head=True,
    )
    footprint = torch.ones(1, 32, 32)
    sampled, trajectory = sample_floorplans(
        model,
        footprint,
        steps=16,
        semantic_decode_weight=1.0,
    )
    assert sampled.shape == (1, 12, 32, 32)
    assert trajectory.shape[1] == 1


def test_advanced_stream_model_shape() -> None:
    model = FloorplanFlowModel(
        target_channels=19,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        topology_aux=False,
        advanced_streams=True,
    )
    x_t = torch.randn(1, 19, 32, 32)
    footprint = torch.ones(1, 32, 32)
    t = torch.tensor([0.5])
    assert model(x_t, t, footprint).shape == x_t.shape


def test_room_partition_auxiliary_loss_backpropagates() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Bath": 10}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    channels = taxonomy.num_target_channels
    model = FloorplanFlowModel(
        target_channels=channels,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        topology_aux=False,
        semantic_endpoint_head=True,
        room_partition_head=True,
        num_room_classes=len(taxonomy.room_ids),
    )
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 5:27, 5:16] = taxonomy.class_id("room:Bedroom")
    target[:, 5:27, 16:27] = taxonomy.class_id("room:Bath")
    footprint = torch.zeros(1, 32, 32)
    footprint[:, 4:28, 4:28] = 1
    loss, parts = flow_matching_loss(
        model,
        target,
        footprint,
        taxonomy=taxonomy,
        room_partition_weight=0.5,
    )
    assert "room_partition_loss" in parts
    loss.backward()
    assert model.room_partition.net[-1].weight.grad is not None


def test_room_mask_auxiliary_loss_backpropagates() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Bath": 10}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    model = FloorplanFlowModel(
        target_channels=taxonomy.num_target_channels,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        topology_aux=False,
        room_mask_head=True,
        room_partition_head=True,
        num_room_classes=len(taxonomy.room_ids),
        room_mask_queries=4,
        room_mask_resolution=16,
    )
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 5:27, 5:16] = taxonomy.class_id("room:Bedroom")
    target[:, 5:27, 16:27] = taxonomy.class_id("room:Bath")
    footprint = torch.zeros(1, 32, 32)
    footprint[:, 4:28, 4:28] = 1
    loss, parts = flow_matching_loss(
        model,
        target,
        footprint,
        taxonomy=taxonomy,
        room_mask_weight=0.2,
        room_partition_weight=0.1,
    )
    assert "room_mask_loss" in parts
    loss.backward()
    assert model.room_mask_decoder.class_head.weight.grad is not None


def test_structured_object_auxiliary_loss_backpropagates() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter({"Toilet": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    model = FloorplanFlowModel(
        target_channels=taxonomy.num_target_channels,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        topology_aux=False,
        structured_object_head=True,
        semantic_taxonomy=taxonomy,
    )
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 5:27, 5:27] = taxonomy.class_id("room:Bedroom")
    target[:, 5, 5:10] = taxonomy.class_id("wall")
    target[:, 10, 5:8] = taxonomy.opening_ids[0]
    target[:, 20, 20:23] = taxonomy.furniture_ids[0]
    footprint = torch.zeros(1, 32, 32)
    footprint[:, 4:28, 4:28] = 1
    loss, parts = flow_matching_loss(
        model,
        target,
        footprint,
        positive_class_weights=torch.full((taxonomy.num_target_channels,), 5.0),
        taxonomy=taxonomy,
        structured_object_weight=0.2,
    )
    assert "structured_object_loss" in parts
    loss.backward()
    assert model.structured_objects.opening.weight.grad is not None


def test_semantic_anchor_fusion_loss_backpropagates() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    model = FloorplanFlowModel(
        target_channels=taxonomy.num_target_channels,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        topology_aux=False,
        semantic_endpoint_head=True,
        semantic_anchor_fusion=True,
    )
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 5:27, 5:27] = taxonomy.class_id("room:Bedroom")
    footprint = torch.zeros(1, 32, 32)
    footprint[:, 4:28, 4:28] = 1
    loss, parts = flow_matching_loss(
        model,
        target,
        footprint,
        taxonomy=taxonomy,
        semantic_anchor_weight=0.2,
    )
    assert "semantic_anchor_loss" in parts
    assert 0.0 <= float(parts["semantic_anchor_gate"]) <= 1.0
    loss.backward()
    assert model.semantic_anchor_fusion.net[-1].weight.grad is not None


def test_structured_decoder_composes_joint_heads() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter({"Toilet": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    model = FloorplanFlowModel(
        target_channels=taxonomy.num_target_channels,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        topology_aux=False,
        room_mask_head=True,
        room_partition_head=True,
        num_room_classes=len(taxonomy.room_ids),
        room_mask_queries=4,
        room_mask_resolution=16,
        structured_object_head=True,
        semantic_taxonomy=taxonomy,
    ).eval()
    reference = torch.zeros(1, 32, 32, dtype=torch.long)
    footprint = torch.zeros(1, 32, 32)
    footprint[:, 4:28, 4:28] = 1
    labels = decode_structured_floorplan(
        model,
        reference,
        footprint,
        taxonomy,
        room_source="partition",
    )
    assert labels.shape == reference.shape
    assert torch.count_nonzero(labels * (footprint == 0)) == 0
    hybrid, _support, _objects, _decoded = decode_structured_floorplan(
        model,
        reference,
        footprint,
        taxonomy,
        room_source="partition",
        object_source="heads",
        opening_source="heads",
        furniture_source="semantic",
        return_debug=True,
    )
    assert hybrid.shape == reference.shape

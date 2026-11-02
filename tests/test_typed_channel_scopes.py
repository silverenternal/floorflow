from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.training.typed_channel_scopes import (
    configure_typed_channel_scope,
    make_typed_channel_optimizer,
)


def _model(
    *,
    program_router: bool = False,
    action_route: bool = False,
) -> torch.nn.Module:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter({"Sink": 2}),
            "door": Counter({"Door": 2}),
            "window": Counter({"Window": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    structure_channels = (
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3
    )
    return build_predicted_channel_model(
        taxonomy,
        {
            "control_channels": [
                len(taxonomy.room_ids),
                structure_channels,
                len(taxonomy.furniture_ids) + 1,
                2,
            ],
            "program_channels": 2,
            "grid": 8,
            "field_channels": 8,
            "head_channels": 16,
            "heads": 4,
            "layers": 1,
            "attention_grid": 4,
            "full_resolution_output": False,
            "typed_channel_links": True,
            "control_evidence_context": True,
            "typed_source_spatial_router": True,
            "dynamic_channel_queries": True,
            "source_channel_attention": True,
            "dynamic_cross_attention": True,
            "channel_group_adapters": True,
            "cross_attention_lora_rank": 4,
            "spatial_conv_adapter": True,
            "spatial_adapter_bottleneck": 4,
            "dynamic_stage_heads": True,
            "stage_head_hidden": 16,
            "dynamic_stage_links": True,
            "action_route_attention": action_route,
            "enable_instance_room_decoder": True,
            "enable_program_instance_router": program_router,
            "enable_room_adjacency": True,
            "enable_room_query_set": True,
            "room_instance_queries": 4,
            "room_instance_hidden": 16,
            "room_query_set_hidden": 16,
            "room_query_set_pool": 4,
            "room_query_set_layers": 1,
            "room_query_set_routing_attention": False,
        },
    )


def test_cross_attention_scope_freezes_field_and_stage_heads() -> None:
    model = _model()
    report = configure_typed_channel_scope(model, "cross_attention")
    assert report["trainable_parameters"] > 0
    assert model.head.layers[0].cross_attention.in_proj_weight.requires_grad
    assert not model.field.fuse[0].net[0].weight.requires_grad
    assert not model.stage_refiners["walls"].output.weight.requires_grad


def test_cross_attention_lora_scope_freezes_attention_weights() -> None:
    model = _model()
    configure_typed_channel_scope(model, "cross_attention_lora")
    assert model.head.layers[0].cross_attention_lora.query_up.weight.requires_grad
    assert not model.head.layers[0].cross_attention.in_proj_weight.requires_grad


def test_spatial_adapter_scope_only_updates_conv_adapter() -> None:
    model = _model()
    configure_typed_channel_scope(model, "spatial_conv_adapter")
    assert model.head.spatial_adapter.up.weight.requires_grad
    assert not model.head.pixel_projection[0].weight.requires_grad


def test_typed_source_router_scope_freezes_base_fusion() -> None:
    model = _model()
    configure_typed_channel_scope(model, "typed_source_router")
    assert (
        model.field.source_router.family_logits["walls"]
        .weight.requires_grad
    )
    assert (
        model.field.source_router.family_values["openings"]
        .weight.requires_grad
    )
    assert not model.field.fuse[0].net[0].weight.requires_grad


def test_bias_only_scope_matches_bitfit_semantics() -> None:
    model = _model()
    configure_typed_channel_scope(model, "bias_only")
    assert model.head.layers[0].cross_attention.in_proj_bias.requires_grad
    assert not model.head.layers[0].cross_attention.in_proj_weight.requires_grad


def test_opening_adapter_scope_only_updates_opening_path() -> None:
    model = _model()
    configure_typed_channel_scope(model, "opening_adapter")
    assert (
        model.head.channel_group_adapters.adapters["opening"].delta.requires_grad
    )
    assert (
        model.head.channel_group_adapters.adapters["opening"]
        .feature_scale.requires_grad
    )
    assert model.stage_refiners["openings"].output.weight.requires_grad
    assert (
        model.field.source_router.family_logits["openings"]
        .weight.requires_grad
    )
    assert not (
        model.head.channel_group_adapters.adapters["wall"].delta.requires_grad
    )
    assert not model.field.fuse[1].net[0].weight.requires_grad
    assert not (
        model.field.source_router.family_logits["walls"]
        .weight.requires_grad
    )


def test_family_scope_optimizer_does_not_update_frozen_parameters() -> None:
    model = _model()
    configure_typed_channel_scope(model, "opening_adapter")
    frozen = model.field.fuse[0].net[0].weight.detach().clone()
    wall_adapter = (
        model.head.channel_group_adapters.adapters["wall"].delta.detach().clone()
    )
    opening_adapter = (
        model.head.channel_group_adapters.adapters["opening"].delta.detach().clone()
    )
    optimizer = make_typed_channel_optimizer(model, 0.1, 0.0)
    taxonomy = model.taxonomy
    controls = {
        "room_logits": torch.randn(1, len(taxonomy.room_ids), 8, 8),
        "structure": torch.randn(
            1,
            len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
            8,
            8,
        ),
        "furniture": torch.randn(
            1,
            len(taxonomy.furniture_ids) + 1,
            8,
            8,
        ),
        "support": torch.randn(1, 2, 8, 8),
        "program": torch.tensor([[1, 0]], dtype=torch.bool),
    }
    output = model(controls, torch.ones(1, 8, 8))
    output["stage_openings_logits"].square().mean().backward()
    optimizer.step()
    assert torch.equal(model.field.fuse[0].net[0].weight, frozen)
    assert torch.equal(
        model.head.channel_group_adapters.adapters["wall"].delta,
        wall_adapter,
    )
    assert not torch.equal(
        model.head.channel_group_adapters.adapters["opening"].delta,
        opening_adapter,
    )


def test_room_graph_scope_preserves_decoder_and_wall_handoff() -> None:
    model = _model(program_router=True)
    report = configure_typed_channel_scope(model, "room_graph_adapter")
    stage = model.room_instance_stage
    assert 0 < report["trainable_elements"]
    assert stage.room_adjacency_head.projection.weight.requires_grad
    assert stage.room_graph_conditioner.mask_residual.weight.requires_grad
    assert stage.query_set_head.count[-1].weight.requires_grad
    assert stage.program_router is not None
    assert stage.program_router.query_projection.weight.requires_grad
    assert stage.program_router.scale.requires_grad
    assert not stage.decoder.layers[0].cross_attention.in_proj_weight.requires_grad
    assert not stage.structure_adapter[0].weight.requires_grad
    assert not model.stage_refiners["walls"].output.weight.requires_grad


def test_room_reference_parameters_receive_a_scaled_optimizer_group() -> None:
    model = torch.nn.Module()
    stage = torch.nn.Module()
    decoder = torch.nn.Module()
    decoder.adaptive_query_refiner = torch.nn.Linear(2, 2)
    stage.decoder = decoder
    model.room_instance_stage = stage
    model.base = torch.nn.Parameter(torch.ones(1))
    optimizer = make_typed_channel_optimizer(
        model,
        0.1,
        0.0,
        room_reference_learning_rate_multiplier=10.0,
    )
    learning_rates = sorted(group["lr"] for group in optimizer.param_groups)
    assert learning_rates == [0.1, 1.0]


def test_room_adapter_trains_program_action_routing_controls() -> None:
    model = _model(action_route=True)
    configure_typed_channel_scope(model, "room_adapter")
    graph = model.action_graph
    assert graph.route_budget["program"][-1].weight.requires_grad
    assert graph.route_gate["program"][-1].weight.requires_grad
    assert graph.route_scale is not None
    assert graph.route_scale.requires_grad
    assert graph.dynamic_scale is not None
    assert graph.dynamic_scale.requires_grad
    assert model.field.evidence_fusion[-1].weight.requires_grad

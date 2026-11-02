from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.masked_channel_distribution import (
    MaskedChannelDistributionHead,
    TypedChannelDistributionHead,
)
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
    predicted_channel_model_kwargs,
)
from floorflow.model.predicted_channel_model import PredictedChannelModel
from floorflow.model.typed_channel_graph import (
    OPENING,
    ROOM,
    WALL,
    channel_family_ids,
    typed_channel_prior,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Bath": 10}),
            "furniture": Counter({"Bed": 6}),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_typed_prior_is_nonuniform_and_family_aware() -> None:
    taxonomy = _taxonomy()
    prior = typed_channel_prior(taxonomy)
    families = channel_family_ids(taxonomy)
    room = int((families == ROOM).nonzero()[0])
    wall = int((families == WALL).nonzero()[0])
    opening = int((families == OPENING).nonzero()[0])
    assert prior.shape == (taxonomy.num_classes, taxonomy.num_classes)
    assert prior[room, wall] > prior[room, opening]
    assert prior[wall, opening] > prior[room, wall]
    assert not torch.allclose(prior, prior[0, 0].expand_as(prior))


def test_typed_channel_head_has_channel_weights_and_gradients() -> None:
    taxonomy = _taxonomy()
    features = torch.randn(2, 16, 32, 32, requires_grad=True)
    dense = MaskedChannelDistributionHead(
        16,
        taxonomy.num_classes,
        hidden_channels=32,
        heads=4,
        layers=2,
        grid=4,
    )
    typed = TypedChannelDistributionHead(
        16,
        taxonomy,
        hidden_channels=32,
        heads=4,
        layers=2,
        grid=4,
        full_resolution_output=True,
    )
    dense_output = dense(features)
    typed_output = typed(features)
    assert typed_output["logits"].shape[1] == taxonomy.num_classes
    assert typed_output["logits"].shape[-2:] == (32, 32)
    assert typed_output["channel_reliability"].shape == (2, taxonomy.num_classes)
    assert torch.all(typed_output["channel_reliability"] >= 0.0)
    assert not torch.allclose(dense_output["logits"], typed_output["logits"])
    loss = typed_output["logits"].square().mean()
    loss.backward()
    assert typed.typed_channel_residual.grad is not None
    assert typed.typed_channel_scale.grad is not None
    assert typed.reliability_head[-1].weight.grad is not None


def test_dynamic_typed_channel_graph_is_sample_conditioned_and_trainable() -> None:
    taxonomy = _taxonomy()
    head = TypedChannelDistributionHead(
        8,
        taxonomy,
        hidden_channels=16,
        heads=4,
        layers=1,
        grid=4,
        full_resolution_output=False,
        dynamic_channel_links=True,
    )
    first = head(torch.randn(2, 8, 8, 8))
    second = head(torch.randn(2, 8, 8, 8))
    assert first["channel_attention_bias"].shape == (
        2,
        taxonomy.num_classes,
        taxonomy.num_classes,
    )
    assert not torch.allclose(
        first["channel_attention_bias"],
        second["channel_attention_bias"],
    )
    first["logits"].square().mean().backward()
    assert head.dynamic_channel_scale.grad is not None
    assert head.dynamic_channel_query.weight.grad is not None
    assert head.dynamic_channel_key.weight.grad is not None


def test_source_conditioned_queries_and_cross_attention_are_dynamic() -> None:
    taxonomy = _taxonomy()
    head = TypedChannelDistributionHead(
        8,
        taxonomy,
        hidden_channels=16,
        heads=4,
        layers=1,
        grid=4,
        full_resolution_output=False,
        dynamic_channel_queries=True,
        source_attention=True,
        dynamic_cross_attention=True,
        channel_group_adapters=True,
        cross_attention_lora_rank=4,
        spatial_conv_adapter=True,
        spatial_adapter_bottleneck=4,
    )
    features = torch.randn(2, 8, 8, 8)
    first_sources = torch.randn(2, 4, 8)
    second_sources = torch.randn(2, 4, 8)
    first = head(features, source_tokens=first_sources)
    second = head(features, source_tokens=second_sources)
    assert first["channel_source_attention"].shape == (
        2,
        taxonomy.num_classes,
        4,
    )
    assert first["channel_source_attention_gate"].shape == (
        2,
        taxonomy.num_classes,
        1,
    )
    assert torch.isfinite(first["channel_cross_attention_gate"])
    assert set(first["channel_group_adapter_gates"]) == {
        "room",
        "wall",
        "opening",
        "furniture",
    }
    assert not torch.allclose(
        first["channel_source_attention"],
        second["channel_source_attention"],
    )
    first["logits"].square().mean().backward()
    assert torch.count_nonzero(
        head.dynamic_query_adapter.hyper[0].weight.grad
    ) > 0
    assert torch.count_nonzero(head.source_adapter.query.weight.grad) > 0
    assert torch.count_nonzero(head.source_adapter.gate[0].weight.grad) > 0
    assert torch.count_nonzero(
        head.layers[0].cross_gate[0].weight.grad
    ) > 0
    assert torch.count_nonzero(
        head.layers[0].cross_attention_lora.query_down.weight.grad
    ) > 0
    assert torch.count_nonzero(head.spatial_adapter.down.weight.grad) > 0
    assert torch.count_nonzero(head.reliability_head[0].weight.grad) > 0
    assert torch.count_nonzero(
        head.channel_group_adapters.adapters["room"]
        .source_gate[1].weight.grad
    ) > 0
    assert head.layers[0].cross_attention_lora.rank_logits.grad is not None


def test_control_evidence_context_reaches_channel_queries() -> None:
    taxonomy = _taxonomy()
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(2, 6, 2, 2),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=False,
        typed_channel_links=True,
        control_evidence_context=True,
        dynamic_channel_links=True,
        dynamic_channel_queries=True,
        source_channel_attention=True,
        dynamic_cross_attention=True,
    )
    controls = {
        "room_logits": torch.randn(2, 2, 16, 16),
        "structure": torch.randn(2, 6, 16, 16),
        "furniture": torch.randn(2, 2, 16, 16),
        "support": torch.randn(2, 2, 16, 16),
        "program": torch.tensor([[1, 0], [0, 1]], dtype=torch.bool),
    }
    output = model(controls, torch.ones(2, 16, 16))
    assert output["channel_attention_bias"] is not None
    assert output["channel_source_attention"].shape == (
        2,
        taxonomy.num_classes,
        6,
    )
    assert torch.isfinite(output["channel_cross_attention_gate"])
    output["logits"].square().mean().backward()
    assert model.head.context_projection[-1].weight.grad is not None
    assert model.field.evidence_fusion[-1].weight.grad is not None
    assert model.field.envelope[0].weight.grad is not None
    assert model.head.dynamic_query_adapter.hyper[0].weight.grad is not None


def test_dynamic_stage_heads_are_sample_conditioned_and_trainable() -> None:
    taxonomy = _taxonomy()
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(2, 6, 2, 2),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=False,
        typed_channel_links=True,
        use_action_graph=True,
        control_evidence_context=True,
        dynamic_channel_links=True,
        dynamic_stage_heads=True,
        stage_head_hidden=16,
        dynamic_stage_links=True,
        action_route_attention=True,
        action_route_router_count=4,
        action_route_top_k=2,
        stage_route_attention=True,
        stage_route_grid=4,
        stage_route_router_count=4,
        stage_route_top_k=2,
        stage_route_heads=4,
    )
    first = {
        "room_logits": torch.randn(2, 2, 16, 16),
        "structure": torch.randn(2, 6, 16, 16),
        "furniture": torch.randn(2, 2, 16, 16),
        "support": torch.randn(2, 2, 16, 16),
        "program": torch.tensor([[1, 0], [0, 1]], dtype=torch.bool),
    }
    with torch.no_grad():
        model.stage_refiners["walls"].gate[-1].weight.normal_(0.0, 0.01)
        model.stage_refiners["walls"].token_gate.weight.normal_(0.0, 0.01)
        model.action_graph.update_scale.fill_(0.1)
        model.action_graph.route_scale.fill_(0.1)
    second = {key: value.clone() for key, value in first.items()}
    second["structure"] = torch.randn_like(second["structure"])
    with torch.no_grad():
        first_output = model(first, torch.ones(2, 16, 16))
        second_output = model(second, torch.ones(2, 16, 16))
    assert first_output["stage_walls_spatial_gate"].shape == (2, 8, 8)
    assert model.stage_refiners["rooms"].route is None
    assert model.stage_refiners["walls"].route is not None
    assert (
        model.stage_refiners["openings"].condition_projection.in_channels
        == 4
    )
    assert first_output["stage_openings_spatial_gate"].shape == (2, 8, 8)
    assert first_output["action_route_diagnostics"] is not None
    assert torch.isfinite(
        first_output["action_route_diagnostics"]["walls"]["routing_entropy"]
    )
    wall_routes = first_output["stage_walls_route_diagnostics"]
    assert torch.isfinite(wall_routes["routing_entropy"])
    assert torch.isfinite(wall_routes["route_gate_mean"])
    assert torch.isfinite(wall_routes["route_budget_soft_mean"])
    assert "envelope" not in model.action_graph.route_budget
    assert "envelope" not in model.action_graph.route_gate
    assert not torch.allclose(
        first_output["stage_walls_spatial_gate"],
        second_output["stage_walls_spatial_gate"],
    )
    trained_output = model(first, torch.ones(2, 16, 16))
    trained_output["stage_walls_logits"].square().mean().backward()
    wall_refiner = model.stage_refiners["walls"]
    assert torch.count_nonzero(wall_refiner.output.weight.grad) > 0
    assert torch.count_nonzero(wall_refiner.gate[0].weight.grad) > 0
    assert torch.count_nonzero(wall_refiner.token_gate.weight.grad) > 0
    assert torch.count_nonzero(wall_refiner.route_output.weight.grad) > 0
    assert torch.count_nonzero(wall_refiner.route_budget[0].weight.grad) > 0
    assert model.action_graph.dynamic_scale.grad is not None
    assert model.action_graph.route_budget["walls"][0].weight.grad is not None
    assert model.action_graph.route_gate["walls"][-1].weight.grad is not None


def test_junction_probability_is_a_distinct_action_token() -> None:
    taxonomy = _taxonomy()
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(2, 6, 2, 2),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=True,
        typed_channel_links=True,
        use_action_graph=True,
    )
    controls = {
        "room_logits": torch.randn(1, 2, 16, 16),
        "structure": torch.randn(1, 6, 16, 16),
        "furniture": torch.randn(1, 2, 16, 16),
        "support": torch.randn(1, 2, 16, 16),
        "program": torch.tensor([[1, 0]], dtype=torch.bool),
    }
    footprint = torch.ones(1, 16, 16)
    first = dict(controls, junction_probability=torch.zeros(1, 16, 16))
    second = dict(controls, junction_probability=torch.ones(1, 16, 16))
    with torch.no_grad():
        first_output = model(first, footprint)
        second_output = model(second, footprint)
    assert not torch.allclose(
        first_output["action_tokens"]["junctions"],
        second_output["action_tokens"]["junctions"],
    )


def test_furniture_query_stage_is_connected_to_action_graph(
    monkeypatch,
) -> None:
    taxonomy = _taxonomy()
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(2, 6, 2, 2),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=True,
        typed_channel_links=True,
        use_action_graph=True,
        enable_furniture_queries=True,
        furniture_query_hidden=16,
        furniture_max_queries=4,
        furniture_min_queries=2,
    )
    controls = {
        "room_logits": torch.randn(1, 2, 16, 16),
        "structure": torch.randn(1, 6, 16, 16),
        "furniture": torch.randn(1, 2, 16, 16),
        "support": torch.randn(1, 2, 16, 16),
        "program": torch.tensor([[1, 0]], dtype=torch.bool),
    }
    captured, original = {}, model.furniture_stage.forward

    def capture(*args, **kwargs):
        captured["room_logits"] = kwargs["room_logits"]; return original(*args, **kwargs)

    monkeypatch.setattr(model.furniture_stage, "forward", capture)
    output = model(controls, torch.ones(1, 16, 16))
    assert output["furniture_query_mask_logits"].shape == (1, 4, 8, 8)
    assert output["furniture_query_budget"].shape == (1,)
    assert torch.equal(
        captured["room_logits"],
        output["stage_rooms_logits"],
    )
    assert output["validation_token"].shape == (1, 8)
    loss = (
        output["furniture_query_mask_logits"].sum()
        + output["furniture_query_class_logits"].sum()
    )
    loss.backward()
    assert (
        model.furniture_stage.head.room_fusion.weight.grad is not None
    )


def test_learned_room_stage_has_a_live_gradient_path() -> None:
    taxonomy = _taxonomy()
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(2, 6, 2, 2),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=True,
        typed_channel_links=True,
        use_action_graph=True,
        preserve_room_stage=False,
    )
    controls = {
        "room_logits": torch.randn(1, 2, 16, 16),
        "structure": torch.randn(1, 6, 16, 16),
        "furniture": torch.randn(1, 2, 16, 16),
        "support": torch.randn(1, 2, 16, 16),
        "program": torch.tensor([[1, 0]], dtype=torch.bool),
    }
    output = model(controls, torch.ones(1, 16, 16))
    output["stage_rooms_logits"].sum().backward()
    assert model.stage_heads["rooms"].weight.grad is not None
    assert model.stage_heads["rooms"].weight.grad.abs().sum() > 0.0


def test_checkpoint_builder_preserves_old_and_new_config_contracts() -> None:
    taxonomy = _taxonomy()
    old = predicted_channel_model_kwargs(
        {
            "control_channels": [2, 6, 2, 2],
            "program_channels": 2,
            "grid": 8,
            "field_channels": 8,
            "head_channels": 16,
            "heads": 4,
            "layers": 1,
            "attention_grid": 4,
        }
    )
    assert old["enable_furniture_queries"] is False
    assert old["preserve_room_stage"] is True
    assert old["control_evidence_context"] is False
    assert old["action_context_conditioning"] is False
    assert old["typed_source_spatial_router"] is False
    assert old["dynamic_channel_links"] is False
    assert old["dynamic_channel_queries"] is False
    assert old["source_channel_attention"] is False
    assert old["dynamic_cross_attention"] is False
    assert old["channel_group_adapters"] is False
    assert old["cross_attention_lora_rank"] == 0
    assert old["spatial_conv_adapter"] is False
    assert old["spatial_adapter_bottleneck"] == 16
    assert old["source_hyper_hidden"] == 64
    assert old["dynamic_stage_heads"] is False
    assert old["stage_head_hidden"] == 64
    assert old["dynamic_stage_links"] is False
    assert old["action_route_attention"] is False
    assert old["action_route_router_count"] == 8
    assert old["action_route_top_k"] == 4
    assert old["refresh_action_tokens"] is False
    assert old["enable_room_adjacency"] is False
    assert old["stage_route_attention"] is False
    assert old["stage_route_grid"] == 8
    assert old["stage_route_router_count"] == 12
    assert old["stage_route_top_k"] == 6
    model = build_predicted_channel_model(
        taxonomy,
        {
            **old,
            "enable_furniture_queries": True,
            "furniture_query_hidden": 16,
            "furniture_max_queries": 4,
            "furniture_min_queries": 2,
            "furniture_detach_backbone": True,
            "furniture_room_assignment": True,
            "preserve_room_stage": False,
        },
    )
    assert model.furniture_stage is not None
    assert model.furniture_detach_backbone is True
    assert model.furniture_stage.room_assignment is not None
    assert model.preserve_room_stage is False


def test_detached_furniture_query_loss_does_not_reach_wall_backbone() -> None:
    taxonomy = _taxonomy()
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(2, 6, 2, 2),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=True,
        typed_channel_links=True,
        use_action_graph=True,
        enable_furniture_queries=True,
        furniture_query_hidden=16,
        furniture_max_queries=4,
        furniture_min_queries=2,
        furniture_detach_backbone=True,
    )
    controls = {
        "room_logits": torch.randn(1, 2, 16, 16),
        "structure": torch.randn(1, 6, 16, 16),
        "furniture": torch.randn(1, 2, 16, 16),
        "support": torch.randn(1, 2, 16, 16),
        "program": torch.tensor([[1, 0]], dtype=torch.bool),
    }
    output = model(controls, torch.ones(1, 16, 16))
    query_loss = (
        output["furniture_query_mask_logits"].square().mean()
        + output["furniture_query_class_logits"].square().mean()
    )
    query_loss.backward()
    assert model.field.fuse[0].net[0].weight.grad is None
    assert model.stage_heads["walls"].weight.grad is None
    assert model.furniture_stage.head.decoder.class_head.weight.grad is not None


def test_instance_room_decoder_exposes_typed_room_and_structure_context() -> None:
    taxonomy = _taxonomy()
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(2, 6, 2, 2),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=True,
        typed_channel_links=True,
        use_action_graph=True,
        enable_instance_room_decoder=True,
        enable_room_adjacency=True,
        room_instance_hidden=16,
        room_instance_queries=4,
        room_instance_layers=1,
        enable_program_instance_router=True,
    )
    controls = {
        "room_logits": torch.randn(1, 2, 16, 16),
        "structure": torch.randn(1, 6, 16, 16),
        "furniture": torch.randn(1, 2, 16, 16),
        "support": torch.randn(1, 2, 16, 16),
        "program": torch.tensor([[1, 0]], dtype=torch.bool),
    }
    output = model(controls, torch.ones(1, 16, 16))
    assert output["room_instance_logits"].shape == (1, 2, 8, 8)
    assert output["room_instance_class_logits"].shape == (1, 4, 3)
    assert output["room_instance_mask_logits"].shape == (1, 4, 8, 8)
    assert output["room_adjacency_logits"].shape == (1, 4, 4)
    assert output["slot_owner_probability"].shape == (1, 4, 8, 8)
    assert torch.all(
        output["slot_owner_probability"].sum(dim=1) <= 1.0
    )
    assert output["program_room_assignment_logits"].shape == (
        1,
        4,
        len(taxonomy.room_ids) + 1,
    )
    assert output["program_room_class_logits"].shape == (
        1,
        4,
        len(taxonomy.room_ids) + 1,
    )
    assert torch.equal(
        output["program_room_class_logits"][..., :-1],
        output["program_room_assignment_logits"][..., 1:],
    )
    assert torch.equal(
        output["program_room_class_logits"][..., -1],
        output["program_room_assignment_logits"][..., 0],
    )
    loss = (
        output["room_instance_mask_logits"].sum()
        + output["room_instance_class_logits"].sum()
    )
    loss.backward()
    assert (
        model.room_instance_stage.decoder.class_head.weight.grad is not None
    )

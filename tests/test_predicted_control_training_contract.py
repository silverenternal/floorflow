from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.model.predicted_channel_model import PredictedChannelModel
from floorflow.objectives.cad_stage_loss import cad_stage_loss
from floorflow.objectives.gradient_balancer import GradientNormBalancer
from floorflow.objectives.routing_regularization import (
    typed_routing_regularization,
)
from floorflow.training.room_capability_training import (
    _room_class_weights,
    instance_room_losses,
)
from floorflow.training.typed_channel_cli import _load_experiment_config
from floorflow.training.typed_channel_route_metrics import (
    typed_channel_module_gradient_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def test_room_class_weights_prioritize_rare_taxonomy_types() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 20, "Kitchen": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    weights = _room_class_weights(taxonomy, torch.device("cpu"))
    bedroom = taxonomy.room_ids.index(taxonomy.class_id("room:Bedroom"))
    kitchen = taxonomy.room_ids.index(taxonomy.class_id("room:Kitchen"))
    assert weights[bedroom] < weights[kitchen]


def test_area_fidelity_configs_change_only_area_weight() -> None:
    pairs = (
        (
            "configs/predicted_channel_v6_room_identity_adapter_control.json",
            "configs/predicted_channel_v6_area_fidelity_only.json",
        ),
        (
            "configs/predicted_channel_v6_room_graph_area_preserving.json",
            "configs/predicted_channel_v6_area_fidelity.json",
        ),
    )
    for control_name, candidate_name in pairs:
        control = _load_experiment_config(ROOT / control_name)
        candidate = _load_experiment_config(ROOT / candidate_name)
        assert control["model"] == candidate["model"]
        assert control["train_scope"] == candidate["train_scope"]
        assert candidate["loss_policy"]["room_instance_mask_area_weight"] == 0.2
        control_loss = dict(control.get("loss_policy", {}))
        candidate_loss = dict(candidate["loss_policy"])
        assert candidate_loss.pop("room_instance_mask_area_weight") == 0.2
        control_loss.pop("room_instance_mask_area_weight", None)
        assert control_loss == candidate_loss


def test_relative_owner_config_changes_only_owner_mode() -> None:
    control = _load_experiment_config(
        ROOT
        / "configs/predicted_channel_v6_room_identity_adapter_control.json"
    )
    candidate = _load_experiment_config(
        ROOT / "configs/predicted_channel_v6_owner_relative_only.json"
    )
    control_model = dict(control["model"])
    candidate_model = dict(candidate["model"])
    assert candidate_model.pop("owner_preserving_room_readout") == "relative"
    assert control_model.pop("owner_preserving_room_readout") is True
    assert control_model == candidate_model
    assert control["loss_policy"] == candidate["loss_policy"]


def test_predicted_control_joint_training_contract_backpropagates() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Kitchen": 4}),
            "furniture": Counter({"Bed": 4}),
            "door": Counter({"Swing": 4}),
            "window": Counter({"Regular": 4}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(
            len(taxonomy.room_ids),
            len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
            len(taxonomy.furniture_ids) + 1,
            2,
        ),
        program_channels=len(taxonomy.room_ids),
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=False,
        use_action_graph=True,
        control_evidence_context=True,
        action_context_conditioning=True,
        typed_source_spatial_router=True,
        refresh_action_tokens=True,
        enable_instance_room_decoder=True,
        room_instance_hidden=16,
        room_instance_queries=4,
        room_instance_layers=1,
        room_instance_adaptive_query=True,
        room_instance_adaptive_query_residual_scale=0.05,
        enable_room_adjacency=True,
        room_instance_mask_aware_adjacency=True,
        room_instance_relational_adjacency=True,
        room_instance_class_count_prior=True,
        relational_adjacency_hidden=16,
        room_instance_count_aware_graph=True,
        count_aware_second_graph=True,
        enable_room_query_set=True,
        room_query_set_hidden=16,
        room_query_set_pool=8,
        room_query_set_layers=1,
        room_query_set_routing_attention=True,
        enable_program_instance_router=True,
        program_room_action_fusion=True,
        enable_furniture_queries=True,
        furniture_query_hidden=16,
        furniture_max_queries=4,
        furniture_min_queries=2,
        furniture_room_assignment=True,
        dynamic_stage_heads=True,
        stage_head_hidden=16,
        dynamic_stage_links=True,
        action_route_attention=True,
        action_route_router_count=4,
        action_route_top_k=2,
        differentiable_route_budget=True,
    ).to(device)
    controls = {
        "room_logits": torch.randn(
            1,
            len(taxonomy.room_ids),
            16,
            16,
            device=device,
        ),
        "structure": torch.randn(
            1,
            len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
            16,
            16,
            device=device,
        ),
        "furniture": torch.randn(
            1,
            len(taxonomy.furniture_ids) + 1,
            16,
            16,
            device=device,
        ),
        "support": torch.randn(1, 2, 16, 16, device=device),
        "program": torch.zeros(
            1,
            len(taxonomy.room_ids),
            dtype=torch.bool,
            device=device,
        ),
    }
    controls["program"][0, :2] = True
    room_count = len(taxonomy.room_ids)
    opening_count = len(taxonomy.opening_ids)
    controls["structure"][:, room_count] = torch.sigmoid(controls["structure"][:, room_count])
    controls["structure"][:, room_count + 1] = torch.sigmoid(controls["structure"][:, room_count + 1])
    controls["structure"][
        :,
        room_count + 2 : room_count + 2 + opening_count,
    ] = controls["structure"][
        :,
        room_count + 2 : room_count + 2 + opening_count,
    ].softmax(dim=1)
    controls["structure"][:, room_count + 2 + opening_count] = 1.0
    controls["furniture"] = controls["furniture"].softmax(dim=1)
    footprint = torch.ones(1, 16, 16, device=device, dtype=torch.bool)
    target = torch.zeros(1, 16, 16, dtype=torch.long, device=device)
    target[:, 1:-1, 1:8] = taxonomy.room_ids[0]
    target[:, 1:-1, 8:-1] = taxonomy.room_ids[1]
    target[:, 0] = taxonomy.class_id("wall")
    target[:, -1] = taxonomy.class_id("wall")
    target[:, 3:5, 3:5] = taxonomy.opening_ids[0]
    target[:, 10:12, 10:12] = taxonomy.furniture_ids[0]
    output = model(controls, footprint)
    nonfinite = [
        key for key, value in output.items() if isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
    ]
    assert not nonfinite, nonfinite
    assert "room_graph_second_class_residual" in output
    assert "room_graph_second_mask_residual" in output
    output["query_count_selection_mask"] = torch.zeros_like(
        output["query_count_selection_mask"],
        dtype=torch.float32,
    )
    output["query_count_selection_mask"][:, :2] = 1.0
    instance_loss, partition_loss, _instance_components, loss_terms = instance_room_losses(
        output,
        target,
        footprint,
        taxonomy,
        max_rooms=4,
        program=controls["program"],
        dense_type_adjacency_weight=0.1,
        dense_type_node_weight=0.05,
        dense_type_edge_overlap_weight=0.5,
        dense_type_node_overlap_weight=0.25,
        query_separation_weight=0.1,
        query_separation_class_aware=False,
        mask_area_weight=0.2,
        return_loss_terms=True,
    )
    assert _instance_components["instance_room_query_separation"] > 0.0
    assert _instance_components["instance_room_area"] > 0.0
    assert {
        "instance_room",
        "query_separation",
        "room_area",
        "program_instance",
        "room_adjacency",
        "room_type_adjacency",
        "room_type_node",
        "dense_type_adjacency",
        "dense_type_node",
        "query_existence",
        "query_count",
        "query_class_count",
        "query_class_cardinality",
        "query_slot_class_anchor",
        "query_support_assignment",
        "query_selection_alignment",
        "query_presence_rank",
        "query_presence_count",
    } == set(loss_terms)
    _, _, _, disabled_dense_terms = instance_room_losses(
        output,
        target,
        footprint,
        taxonomy,
        max_rooms=4,
        program=controls["program"],
        return_loss_terms=True,
    )
    assert "dense_type_adjacency" not in disabled_dense_terms
    assert "dense_type_node" not in disabled_dense_terms
    stage_loss, _stage_components, stage_terms = cad_stage_loss(
        output["logits"],
        target,
        footprint,
        taxonomy,
        return_loss_terms=True,
    )
    routing_loss, _routing_components = typed_routing_regularization(
        model,
        output,
    )
    for name, component in (
        ("instance", instance_loss),
        ("partition", partition_loss),
        ("stage", stage_loss),
        ("routing", routing_loss),
    ):
        assert torch.isfinite(component), name
    balancing_terms = {**loss_terms, **stage_terms}
    balancer = GradientNormBalancer(
        {name: 1.0 for name in balancing_terms},
        reference="stage_wall",
        update_interval=1,
    )
    balancer.maybe_update(
        balancing_terms,
        list(model.parameters()),
        step=1,
    )
    touched = balancer.diagnostics()["last_active_parameters"]
    for name in (
        "stage_wall",
        "stage_wall_precision",
        "stage_opening",
        "stage_opening_precision",
        "stage_opening_host",
        "stage_opening_type",
        "stage_opening_family",
        "dense_type_adjacency",
        "dense_type_node",
        "query_existence",
        "query_count",
        "query_class_count",
        "query_class_cardinality",
        "query_separation",
        "room_area",
    ):
        assert touched[name] > 0, name
    loss = (
        instance_loss
        + 0.1 * partition_loss
        + 0.1 * stage_loss
        + 0.01 * routing_loss
        + 0.01 * output["furniture_query_class_logits"].square().mean()
        + 0.01 * output["furniture_query_mask_logits"].square().mean()
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.room_instance_stage.query_set_head.count[-1].weight.grad is not None
    assert model.room_instance_stage.room_adjacency_head.projection.weight.grad is not None
    assert model.room_instance_stage.room_graph_conditioner.scale.grad is not None
    assert model.room_instance_stage.room_graph_conditioner.mask_residual.weight.grad is not None
    assert model.room_instance_stage.room_graph_conditioner.class_residual.weight.grad is not None
    assert model.furniture_stage.head.decoder.class_head.weight.grad is not None
    assert model.action_graph.update_scale.grad is not None
    program_budget = model.action_graph.route_budget["program"][-1].weight.grad
    program_gate = model.action_graph.route_gate["program"][-1].weight.grad
    program_budget_hidden = model.action_graph.route_budget["program"][0].weight.grad
    program_gate_hidden = model.action_graph.route_gate["program"][0].weight.grad
    assert program_budget is not None
    assert program_gate is not None
    assert torch.count_nonzero(program_budget) > 0
    assert torch.count_nonzero(program_gate) > 0
    assert torch.count_nonzero(program_budget_hidden) > 0
    assert torch.count_nonzero(program_gate_hidden) > 0
    assert torch.count_nonzero(model.room_instance_stage.query_set_head.route_budget[0][0].weight.grad) > 0
    assert torch.count_nonzero(model.room_instance_stage.structure_adapter[0].weight.grad) > 0
    assert torch.count_nonzero(model.room_instance_stage.program_router.background.weight.grad) > 0
    assert torch.count_nonzero(model.field.evidence_fusion[0].weight.grad) > 0
    module_gradients = typed_channel_module_gradient_metrics(model)
    for name in (
        "field_evidence_fusion_gradient_norm",
        "room_structure_adapter_gradient_norm",
        "room_query_embedding_gradient_norm",
        "room_reference_head_gradient_norm",
        "room_reference_context_gradient_norm",
        "room_program_class_fusion_gradient_norm",
        "room_query_route_budget_0_gradient_norm",
        "action_program_route_budget_gradient_norm",
        "action_program_route_gate_gradient_norm",
        "action_program_projection_gradient_norm",
        "action_program_gate_gradient_norm",
    ):
        assert module_gradients[name] > 0, name
    assert torch.isfinite(_instance_components["dense_type_edge_loss"])
    assert torch.isfinite(_instance_components["dense_type_node_loss"])

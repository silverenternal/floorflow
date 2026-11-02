from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from floorflow.data.taxonomy import build_taxonomy
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.training.typed_channel_cli import model_config, parse_args
from floorflow.training.typed_channel_instance_loss_kwargs import (
    typed_channel_instance_loss_kwargs,
)
from floorflow.training.typed_channel_loss_balance import (
    build_typed_channel_balancer,
)


def test_typed_channel_defaults_have_a_real_training_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--output-dir", "outputs/train/test"],
    )
    args = parse_args()
    assert args.predicted_root.name == "model_predicted_full_v3"
    assert args.limit_train == 512
    assert args.epochs == 16
    assert args.field_channels == 64
    assert args.head_channels == 128
    assert args.attention_grid == 24
    assert args.layers == 3
    assert args.control_evidence_context
    assert args.action_context_conditioning
    assert args.enable_instance_room_decoder
    assert args.enable_room_adjacency
    assert args.enable_room_query_set
    assert args.enable_furniture_queries
    assert args.furniture_room_assignment
    assert args.room_instance_hidden == 128
    assert args.room_query_set_hidden == 96
    assert args.furniture_query_hidden == 128
    assert args.furniture_max_queries == 32
    assert args.dynamic_channel_links
    assert args.dynamic_channel_queries
    assert args.source_channel_attention
    assert args.dynamic_cross_attention
    assert args.channel_group_adapters
    assert args.cross_attention_lora_rank == 8
    assert args.spatial_conv_adapter
    assert args.spatial_adapter_bottleneck == 16
    assert args.train_scope == "full"
    assert args.source_hyper_hidden == 64
    assert args.dynamic_stage_heads
    assert args.stage_head_hidden == 128
    assert args.action_route_attention
    assert args.action_route_router_count == 8
    assert args.action_route_top_k == 4
    assert args.stage_route_attention
    assert args.stage_route_grid == 8
    assert args.stage_route_router_count == 12
    assert args.stage_route_top_k == 6
    assert args.dynamic_balance_update_interval == 20
    assert args.dynamic_balance_exponent == 1.0
    assert not args.room_instance_class_balance


def test_dynamic_channel_model_is_within_the_declared_parameter_budget() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Bath": 10}),
            "furniture": Counter({"Bed": 4}),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter({"Regular": 4}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    model = build_predicted_channel_model(
        taxonomy,
        {
            "control_channels": [2, 6, 2, 2],
            "program_channels": 2,
            "grid": 8,
            "field_channels": 64,
            "head_channels": 128,
            "heads": 4,
            "layers": 3,
            "attention_grid": 24,
            "full_resolution_output": True,
            "typed_channel_links": True,
            "control_evidence_context": True,
            "dynamic_channel_links": True,
            "dynamic_channel_queries": True,
            "source_channel_attention": True,
            "dynamic_cross_attention": True,
            "channel_group_adapters": True,
            "cross_attention_lora_rank": 8,
            "spatial_conv_adapter": True,
            "spatial_adapter_bottleneck": 8,
            "source_hyper_hidden": 32,
            "dynamic_stage_heads": True,
            "stage_head_hidden": 96,
            "dynamic_stage_links": True,
            "action_route_attention": True,
            "action_route_router_count": 4,
            "action_route_top_k": 2,
            "stage_route_attention": True,
            "stage_route_grid": 4,
            "stage_route_router_count": 4,
            "stage_route_top_k": 3,
        },
    )
    assert model.control_evidence_context
    assert model.dynamic_channel_links
    assert model.dynamic_channel_queries
    assert model.source_channel_attention
    assert model.dynamic_cross_attention
    assert 1_000_000 < sum(p.numel() for p in model.parameters()) < 4_000_000
    assert model.stage_route_attention
    assert model.action_graph is not None
    assert model.action_graph.route is not None


def test_trainer_defaults_match_promoted_model_config(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--output-dir", "outputs/train/test"],
    )
    actual = model_config(parse_args())
    config = json.loads(
        Path("configs/predicted_channel_source_router_v3.json").read_text(
            encoding="utf-8"
        )
    )["model"]
    expected = dict(config)
    expected["grid"] = expected.pop("field_grid")
    expected.pop("estimated_parameters", None)
    expected["control_channels"] = tuple(expected["control_channels"])
    assert {key: actual[key] for key in expected} == expected
    assert not actual["room_instance_relational_adjacency"]
    assert not actual["room_instance_class_count_prior"]
    assert not actual["count_aware_second_graph"]
    assert actual["relational_adjacency_hidden"] == 128


def test_dynamic_loss_balancer_covers_weak_metric_objectives(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--output-dir", "outputs/train/test"],
    )
    args = parse_args()
    args.dynamic_loss_weights = True
    args.dynamic_balance_update_interval = 7
    balancer = build_typed_channel_balancer(args)
    assert balancer is not None
    assert balancer.update_interval == 7
    assert balancer.reference == "instance_room"
    assert set(balancer.base_weights) == {
        "ce",
        "room_iou",
        "tversky",
        "boundary",
        "family_iou",
        "stage_wall",
        "stage_wall_precision",
        "stage_opening",
        "stage_opening_precision",
        "stage_opening_host",
        "stage_opening_type",
        "stage_opening_family",
        "stage_furniture",
        "routing",
        "furniture_query",
        "instance_room",
        "query_separation",
        "room_area",
        "instance_partition",
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
            "query_support_assignment",
        "query_selection_alignment",
        "query_presence_rank",
        "query_presence_count",
        "furniture_assignment",
        "furniture_containment",
    }


def test_v5_model_config_drives_model_and_loss_atomically(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--output-dir",
            "outputs/train/test",
            "--model-config",
            "configs/predicted_channel_source_router_v5_relational_edge.json",
        ],
    )
    args = parse_args()
    resolved = model_config(args)
    assert resolved["room_instance_relational_adjacency"]
    assert resolved["room_instance_class_count_prior"]
    assert resolved["count_aware_second_graph"]
    assert resolved["relational_adjacency_hidden"] == 128
    assert resolved["grid"] == 128
    assert args.dynamic_loss_weights
    assert args.dynamic_balance_exponent == 0.5
    assert args.room_instance_class_balance
    assert args.dense_type_adjacency_weight == 0.1
    assert args.dense_type_node_weight == 0.05
    assert args.dense_type_edge_overlap_weight == 0.5
    assert args.dense_type_node_overlap_weight == 0.25
    assert args.wall_precision_weight == 0.5
    assert args.opening_precision_weight == 0.5
    assert args.opening_host_weight == 0.5
    assert args.room_query_class_cardinality_weight == 0.1


def test_v5_control_inherits_only_declared_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--output-dir",
            "outputs/train/test",
            "--model-config",
            "configs/predicted_channel_source_router_v5_control.json",
        ],
    )
    args = parse_args()
    resolved = model_config(args)
    assert not resolved["room_instance_relational_adjacency"]
    assert not resolved["room_instance_class_count_prior"]
    assert not resolved["count_aware_second_graph"]
    assert resolved["room_instance_mask_aware_adjacency"]
    assert args.dynamic_loss_weights
    assert args.dynamic_balance_exponent == 0.5
    assert not args.room_instance_class_balance
    assert args.dense_type_adjacency_weight == 0.1


def test_room_identity_repair_uses_trainable_room_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--output-dir",
            "outputs/train/test",
            "--model-config",
            "configs/predicted_channel_v6_room_identity_adapter_joint.json",
        ],
    )
    args = parse_args()
    resolved = model_config(args)
    assert args.train_scope == "room_adapter"
    assert args.room_instance_relational_adjacency
    assert args.room_instance_class_count_prior
    assert args.count_aware_second_graph
    assert not args.full_resolution_mask_refinement
    assert args.dynamic_balance_reference == "auto"
    assert args.room_query_support_assignment_weight == 0.1
    assert args.room_instance_unmatched_mask_weight == 0.2
    assert args.room_instance_mask_pos_weight == 2.0
    assert args.room_instance_mask_tversky_weight == 0.2
    assert args.room_instance_query_separation_weight == 0.1
    assert args.room_instance_query_separation_class_agnostic
    assert args.room_instance_reference_anchor_weight == 0.2
    assert args.room_instance_mask_centroid_weight == 0.5
    assert args.room_reference_learning_rate_multiplier == 10.0
    assert resolved["program_room_action_fusion"]
    assert resolved["differentiable_route_budget"]
    loss_kwargs = typed_channel_instance_loss_kwargs(
        args,
        return_loss_terms=True,
    )
    assert loss_kwargs["query_separation_weight"] == 0.1
    assert not loss_kwargs["query_separation_class_aware"]
    balancer = build_typed_channel_balancer(args)
    assert balancer is not None
    assert abs(balancer.base_weights["query_separation"] - 0.05) < 1.0e-12


def test_program_route_candidate_isolates_route_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--output-dir",
            "outputs/train/test",
            "--model-config",
            "configs/predicted_channel_v6_program_route_only.json",
        ],
    )
    args = parse_args()
    resolved = model_config(args)
    assert resolved["program_room_action_fusion"]
    assert resolved["differentiable_route_budget"]
    assert args.room_instance_query_separation_weight == 0.0
    assert not args.room_instance_query_separation_class_agnostic
    assert not args.room_instance_relational_adjacency
    assert not args.room_instance_class_count_prior
    assert not args.count_aware_second_graph


def test_area_preserving_graph_candidate_is_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--output-dir",
            "outputs/train/test",
            "--model-config",
            "configs/predicted_channel_v6_room_graph_area_preserving.json",
        ],
    )
    args = parse_args()
    resolved = model_config(args)
    assert resolved["area_preserving_mask_residual"]
    assert resolved["room_instance_relational_adjacency"]
    assert resolved["count_aware_second_graph"]


@pytest.mark.parametrize(
    ("config", "fusion", "budget"),
    [
        (
            "configs/predicted_channel_v6_program_fusion_only.json",
            True,
            False,
        ),
        (
            "configs/predicted_channel_v6_route_budget_only.json",
            False,
            True,
        ),
    ],
)
def test_program_route_single_factor_configs(
    monkeypatch,
    config: str,
    fusion: bool,
    budget: bool,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--output-dir",
            "outputs/train/test",
            "--model-config",
            config,
        ],
    )
    args = parse_args()
    resolved = model_config(args)
    assert resolved.get("program_room_action_fusion", False) is fusion
    assert resolved.get("differentiable_route_budget", False) is budget
    assert args.room_instance_query_separation_weight == 0.0


def test_v5_room_adapter_config_preserves_prior_stage(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--output-dir",
            "outputs/train/test",
            "--model-config",
            "configs/predicted_channel_v6_room_graph_adapter_joint.json",
        ],
    )
    args = parse_args()
    resolved = model_config(args)
    assert args.train_scope == "room_graph_adapter"
    assert args.lr == 1.0e-4
    assert args.init_checkpoint is not None
    assert args.init_checkpoint.name == "best.pt"
    assert resolved["room_instance_class_count_prior"]
    assert resolved["room_instance_relational_adjacency"]
    assert resolved["count_aware_second_graph"]
    assert args.room_instance_class_balance
    assert args.dynamic_balance_exponent == 0.5

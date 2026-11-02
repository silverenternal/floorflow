"""Canonical construction of typed predicted-channel model checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.predicted_channel_model import PredictedChannelModel


def predicted_channel_model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read every architecture option while preserving old checkpoints."""

    return {
        "control_channels": tuple(
            int(value)
            for value in config.get("control_channels", (24, 38, 52, 28))
        ),
        "program_channels": int(config.get("program_channels", 24)),
        "grid": int(config.get("grid", 64)),
        "field_channels": int(config.get("field_channels", 48)),
        "head_channels": int(config.get("head_channels", 96)),
        "heads": int(config.get("heads", 4)),
        "layers": int(config.get("layers", 2)),
        "attention_grid": int(config.get("attention_grid", 16)),
        "full_resolution_output": bool(
            config.get("full_resolution_output", True)
        ),
        "typed_channel_links": bool(
            config.get("typed_channel_links", True)
        ),
        "use_action_graph": bool(config.get("use_action_graph", True)),
        "preserve_room_stage": bool(
            config.get("preserve_room_stage", True)
        ),
        "enable_furniture_queries": bool(
            config.get("enable_furniture_queries", False)
        ),
        "furniture_query_hidden": int(
            config.get("furniture_query_hidden", 96)
        ),
        "furniture_max_queries": int(
            config.get("furniture_max_queries", 24)
        ),
        "furniture_min_queries": int(
            config.get("furniture_min_queries", 4)
        ),
        "furniture_detach_backbone": bool(
            config.get("furniture_detach_backbone", False)
        ),
        "furniture_room_assignment": bool(
            config.get("furniture_room_assignment", False)
        ),
        "furniture_semantic_router": bool(
            config.get("furniture_semantic_router", False)
        ),
        "furniture_semantic_hidden": int(
            config.get("furniture_semantic_hidden", 64)
        ),
        "enable_instance_room_decoder": bool(
            config.get("enable_instance_room_decoder", False)
        ),
        "room_instance_hidden": int(
            config.get("room_instance_hidden", 96)
        ),
        "room_instance_queries": int(
            config.get("room_instance_queries", 32)
        ),
        "room_instance_layers": int(
            config.get("room_instance_layers", 2)
        ),
        "enable_program_instance_router": bool(
            config.get("enable_program_instance_router", False)
        ),
        "program_room_action_fusion": bool(
            config.get("program_room_action_fusion", False)
        ),
        "enable_room_adjacency": bool(
            config.get("enable_room_adjacency", False)
        ),
        "enable_room_query_set": bool(
            config.get("enable_room_query_set", False)
        ),
        "room_query_set_hidden": int(
            config.get("room_query_set_hidden", 64)
        ),
        "room_query_set_pool": int(
            config.get("room_query_set_pool", 48)
        ),
        "room_query_set_layers": int(
            config.get("room_query_set_layers", 2)
        ),
        "room_query_set_routing_attention": bool(
            config.get("room_query_set_routing_attention", True)
        ),
        "query_count_gate": bool(config.get("query_count_gate", False)),
        "owner_preserving_room_readout": bool(
            config.get("owner_preserving_room_readout", False)
        ),
        "room_instance_adaptive_query": bool(
            config.get("room_instance_adaptive_query", False)
        ),
        "room_instance_adaptive_query_residual_scale": float(
            config.get(
                "room_instance_adaptive_query_residual_scale",
                0.0,
            )
        ),
        "room_instance_mask_aware_adjacency": bool(
            config.get("room_instance_mask_aware_adjacency", False)
        ),
        "room_instance_relational_adjacency": bool(
            config.get("room_instance_relational_adjacency", False)
        ),
        "room_instance_class_count_prior": bool(
            config.get("room_instance_class_count_prior", False)
        ),
        "relational_adjacency_hidden": int(
            config.get("relational_adjacency_hidden", 128)
        ),
        "room_instance_count_aware_graph": bool(
            config.get("room_instance_count_aware_graph", False)
        ),
        "count_aware_second_graph": bool(
            config.get("count_aware_second_graph", False)
        ),
        "full_resolution_mask_refinement": bool(
            config.get("full_resolution_mask_refinement", False)
        ),
        "typed_opening_host_gate": bool(
            config.get("typed_opening_host_gate", False)
        ),
        "control_evidence_context": bool(
            config.get("control_evidence_context", False)
        ),
        "action_context_conditioning": bool(
            config.get("action_context_conditioning", False)
        ),
        "typed_source_spatial_router": bool(
            config.get("typed_source_spatial_router", False)
        ),
        "dynamic_channel_links": bool(
            config.get("dynamic_channel_links", False)
        ),
        "dynamic_channel_queries": bool(
            config.get("dynamic_channel_queries", False)
        ),
        "source_channel_attention": bool(
            config.get("source_channel_attention", False)
        ),
        "dynamic_cross_attention": bool(
            config.get("dynamic_cross_attention", False)
        ),
        "channel_group_adapters": bool(
            config.get("channel_group_adapters", False)
        ),
        "cross_attention_lora_rank": int(
            config.get("cross_attention_lora_rank", 0)
        ),
        "spatial_conv_adapter": bool(
            config.get("spatial_conv_adapter", False)
        ),
        "spatial_adapter_bottleneck": int(
            config.get("spatial_adapter_bottleneck", 16)
        ),
        "source_hyper_hidden": int(
            config.get("source_hyper_hidden", 64)
        ),
        "dynamic_stage_heads": bool(
            config.get("dynamic_stage_heads", False)
        ),
        "stage_head_hidden": int(
            config.get("stage_head_hidden", 64)
        ),
        "dynamic_stage_links": bool(
            config.get("dynamic_stage_links", False)
        ),
        "action_route_attention": bool(
            config.get("action_route_attention", False)
        ),
        "action_route_router_count": int(
            config.get("action_route_router_count", 8)
        ),
        "action_route_top_k": int(
            config.get("action_route_top_k", 4)
        ),
        "differentiable_route_budget": bool(
            config.get("differentiable_route_budget", False)
        ),
        "refresh_action_tokens": bool(
            config.get("refresh_action_tokens", False)
        ),
        "refresh_validation_tokens": bool(
            config.get("refresh_validation_tokens", False)
        ),
        "stage_route_attention": bool(
            config.get("stage_route_attention", False)
        ),
        "stage_route_grid": int(config.get("stage_route_grid", 8)),
        "stage_route_router_count": int(
            config.get("stage_route_router_count", 12)
        ),
        "stage_route_top_k": int(config.get("stage_route_top_k", 6)),
        "stage_route_heads": int(config.get("stage_route_heads", 4)),
    }


def build_predicted_channel_model(
    taxonomy: FloorplanTaxonomy,
    config: Mapping[str, Any],
) -> PredictedChannelModel:
    return PredictedChannelModel(
        taxonomy,
        **predicted_channel_model_kwargs(config),
    )


def load_predicted_channel_state_dict(
    model: PredictedChannelModel,
    state: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    """Load old typed checkpoints with explicit identity-preserving migrations."""

    current = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    migrations: list[str] = []
    unsafe_shapes: list[str] = []
    for name, target in current.items():
        source = state.get(name)
        if source is None:
            if name == "action_graph.update_scale":
                compatible[name] = torch.full_like(target, 10.0)
                migrations.append(name)
            elif target.numel() == 1:
                compatible[name] = target.clone()
                migrations.append(name)
            else:
                unsafe_shapes.append(name)
            continue
        if source.shape == target.shape:
            compatible[name] = source
            continue
        if source.numel() == 1 and target.numel() == 1:
            compatible[name] = source.reshape_as(target)
            migrations.append(name)
            continue
        if name == "field.evidence_fusion.0.weight":
            if (
                source.ndim != 2
                or target.ndim != 2
                or source.shape[0] != target.shape[0]
                or source.shape[1] > target.shape[1]
            ):
                unsafe_shapes.append(name)
                continue
            value = torch.zeros_like(target)
            value[:, : source.shape[1]] = source
            compatible[name] = value
            migrations.append(name)
            continue
        if name == "stage_refiners.openings.condition_projection.weight":
            if (
                source.ndim != 4
                or target.ndim != 4
                or source.shape[0] != target.shape[0]
                or source.shape[2:] != target.shape[2:]
                or source.shape[1] != 1
            ):
                unsafe_shapes.append(name)
                continue
            value = torch.zeros_like(target)
            value[:, :1] = source
            compatible[name] = value
            migrations.append(name)
            continue
        unsafe_shapes.append(name)
    if unsafe_shapes:
        raise ValueError(
            f"unsupported checkpoint shape changes: {sorted(unsafe_shapes)}"
        )
    result = model.load_state_dict(compatible, strict=False)
    return {
        "migrations": sorted(migrations),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


__all__ = [
    "build_predicted_channel_model",
    "load_predicted_channel_state_dict",
    "predicted_channel_model_kwargs",
]

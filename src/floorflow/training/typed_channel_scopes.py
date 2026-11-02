"""Granular parameter scopes for fast predicted-channel iteration."""

from __future__ import annotations

import torch
from torch import nn

SCOPES = (
    "full",
    "field",
    "typed_source_router",
    "channel_head",
    "cross_attention",
    "cross_attention_lora",
    "adalora",
    "lora_ranks",
    "spatial_conv_adapter",
    "bias_only",
    "channel_queries",
    "group_adapters",
    "group_ssf",
    "group_delta",
    "stage_heads",
    "mita_routes",
    "room_adapter",
    "room_graph_adapter",
    "wall_adapter",
    "opening_adapter",
    "furniture_adapter",
)


def _matches(name: str, scope: str) -> bool:
    if scope == "full":
        return True
    if scope == "field":
        return name.startswith("field.")
    if scope == "typed_source_router":
        return name.startswith("field.source_router.")
    if scope == "channel_head":
        return name.startswith("head.")
    if scope == "cross_attention":
        return ".cross_attention." in name or ".cross_gate." in name
    if scope == "cross_attention_lora":
        return ".cross_attention_lora." in name
    if scope == "adalora":
        return ".cross_attention_lora." in name
    if scope == "lora_ranks":
        return name.endswith(".rank_logits")
    if scope == "spatial_conv_adapter":
        return ".spatial_adapter." in name
    if scope == "bias_only":
        return name.endswith((".bias", "_bias"))
    if scope == "channel_queries":
        return name == "head.channel_queries" or ".channel_group_adapters." in name
    if scope == "group_adapters":
        return ".channel_group_adapters." in name
    if scope == "group_ssf":
        return ".channel_group_adapters." in name and name.endswith(
            (".feature_scale", ".feature_shift")
        )
    if scope == "group_delta":
        return ".channel_group_adapters." in name and name.endswith(
            (".delta", ".scale", ".source_gate.1.weight", ".source_gate.1.bias")
        )
    if scope == "stage_heads":
        return name.startswith(
            ("stage_refiners.", "stage_scales.", "action_graph.")
        )
    if scope == "mita_routes":
        return ".route." in name or ".route_" in name or name.startswith(
            "action_graph.route"
        )
    if scope == "room_graph_adapter":
        return name.startswith(
            (
                "room_instance_stage.room_adjacency_head.",
                "room_instance_stage.room_graph_conditioner.",
                "room_instance_stage.query_set_head.",
                "room_instance_stage.program_router.",
            )
        ) or name in {
            "room_instance_stage.residual_scale",
            "room_instance_stage.class_count_prior_scale",
            "room_instance_stage.decoder.fusion_scale",
        }
    family = scope.removesuffix("_adapter")
    if scope.endswith("_adapter"):
        stage = "furniture" if family == "furniture" else f"{family}s"
        prefixes = (
            f"head.channel_group_adapters.adapters.{family}.",
            f"stage_refiners.{stage}.",
            f"stage_scales.{stage}",
            f"action_graph.projections.{stage}.",
            f"action_graph.gates.{stage}.",
            f"action_graph.route_budget.{stage}.",
            f"action_graph.route_gate.{stage}.",
            f"field.source_router.family_logits.{stage}.",
            f"field.source_router.family_values.{stage}.",
        )
        if family == "furniture":
            prefixes += ("furniture_stage.",)
        if family == "room":
            prefixes += (
                "room_instance_stage.",
                "action_graph.route_budget.program.",
                "action_graph.route_gate.program.",
                "field.evidence_fusion.2.",
            )
        room_route_controls = family == "room" and name in {
            "action_graph.route_scale",
            "action_graph.dynamic_scale",
            "action_graph.link_residual",
            "action_graph.update_scale",
        }
        return name.startswith(prefixes) or room_route_controls
    return False


def configure_typed_channel_scope(
    model: nn.Module,
    scope: str,
) -> dict[str, object]:
    """Freeze all parameters outside one named scope."""

    if scope not in SCOPES:
        raise ValueError(f"unknown typed-channel train scope: {scope}")
    if scope in {
        "room_adapter",
        "wall_adapter",
        "opening_adapter",
        "furniture_adapter",
    } and not getattr(
        getattr(model, "head", None),
        "channel_group_adapters",
        None,
    ):
        raise ValueError(
            "channel-family scopes require channel_group_adapters"
        )
    selected = []
    for name, parameter in model.named_parameters():
        trainable = _matches(name, scope)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(name)
    if not selected:
        raise RuntimeError(f"train scope {scope!r} selected no parameters")
    return {
        "scope": scope,
        "trainable_parameters": len(selected),
        "trainable_elements": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "parameter_names": selected,
    }


def make_typed_channel_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    *,
    room_reference_learning_rate_multiplier: float = 1.0,
) -> torch.optim.Optimizer:
    reference_prefixes = (
        "room_instance_stage.decoder.adaptive_query_refiner.",
        "room_instance_stage.decoder.query_feature.",
        "room_instance_stage.decoder.query_position.",
    )
    reference = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(reference_prefixes)
    ]
    remaining = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(reference_prefixes)
    ]
    if not reference and not remaining:
        raise RuntimeError("optimizer received no trainable parameters")
    multiplier = max(1.0, float(room_reference_learning_rate_multiplier))
    groups = [{"params": remaining, "lr": float(learning_rate)}]
    if reference:
        groups.append(
            {
                "params": reference,
                "lr": float(learning_rate) * multiplier,
            }
        )
    return torch.optim.AdamW(
        [group for group in groups if group["params"]],
        weight_decay=float(weight_decay),
    )


def make_typed_channel_optimizer_from_args(
    model: nn.Module,
    args,
) -> torch.optim.Optimizer:
    """Construct the trainer optimizer with scoped anchor learning rates."""

    return make_typed_channel_optimizer(
        model,
        float(args.lr),
        float(args.weight_decay),
        room_reference_learning_rate_multiplier=float(
            args.room_reference_learning_rate_multiplier
        ),
    )


__all__ = [
    "SCOPES",
    "configure_typed_channel_scope",
    "make_typed_channel_optimizer",
    "make_typed_channel_optimizer_from_args",
]

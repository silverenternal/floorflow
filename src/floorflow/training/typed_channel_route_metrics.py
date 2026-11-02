"""Flatten stage-MIta diagnostics into validation metric rows."""

from __future__ import annotations

import math

import torch
from torch import nn

STAGES = ("walls", "openings", "furniture")


def typed_channel_route_metrics(
    output: dict[str, object],
) -> dict[str, float]:
    """Return finite scalar route-health metrics without changing model output."""

    pending: dict[str, torch.Tensor] = {}

    def scalar(prefix: str, name: str, value: object) -> None:
        if not isinstance(value, torch.Tensor):
            return
        pending[f"{prefix}_{name}"] = value.detach().float().mean()

    def norm(prefix: str, name: str, value: object) -> None:
        if not isinstance(value, torch.Tensor):
            return
        flattened = value.detach().float().flatten(2)
        pending[f"{prefix}_{name}_l2"] = flattened.square().sum(dim=-1).sqrt().mean()

    def add(prefix: str, diagnostics: object) -> None:
        if not isinstance(diagnostics, dict):
            return
        values = {
            "budget": diagnostics.get("route_budget_soft_mean"),
            "routing_entropy": diagnostics.get("routing_entropy"),
            "attention_entropy": diagnostics.get("attention_entropy"),
            "effective_expert_fraction": diagnostics.get(
                "effective_expert_fraction"
            ),
            "gate_mean": diagnostics.get("route_gate_mean"),
        }
        for name, value in values.items():
            scalar(prefix, name, value)

    scalar(
        "channel_cross_attention",
        "gate_mean",
        output.get("channel_cross_attention_gate"),
    )
    scalar(
        "channel_source_attention",
        "entropy",
        output.get("channel_source_attention_entropy"),
    )
    scalar(
        "channel_source_attention",
        "gate_mean",
        output.get("channel_source_attention_gate"),
    )
    for stage in STAGES:
        add(
            f"stage_{stage}_route",
            output.get(f"stage_{stage}_route_diagnostics"),
        )
    action = output.get("action_route_diagnostics")
    if isinstance(action, dict):
        for stage, diagnostics in action.items():
            add(f"action_{stage}_route", diagnostics)
    scalar("channel_reliability", "mean", output.get("reliability"))
    norm(
        "room_class",
        "mask_region_evidence",
        output.get("mask_class_evidence"),
    )
    for pass_name in ("first", "second"):
        norm(
            f"room_graph_{pass_name}",
            "class_residual",
            output.get(f"room_graph_{pass_name}_class_residual"),
        )
        norm(
            f"room_graph_{pass_name}",
            "mask_residual",
            output.get(f"room_graph_{pass_name}_mask_residual"),
        )
    scalar(
        "room_graph",
        "query_presence_mean",
        output.get("room_graph_query_presence"),
    )
    scalar(
        "room_owner",
        "soft_weight_mean",
        output.get("room_owner_soft_weight_mean"),
    )
    scalar(
        "room_query",
        "selected_count",
        output.get("room_query_selected_count"),
    )
    scalar(
        "room_query",
        "residual_abs_mean",
        output.get("room_query_residual_abs_mean"),
    )
    query_set = output.get("query_set_diagnostics")
    if isinstance(query_set, dict):
        scalar(
            "room_query_set",
            "presence_count_scale",
            query_set.get("presence_count_scale"),
        )
        scalar(
            "room_query_set",
            "presence_expected_count",
            query_set.get("presence_expected_count"),
        )
    adaptive = output.get("adaptive_query_diagnostics")
    if isinstance(adaptive, dict):
        scalar(
            "room_adaptive_reference",
            "displacement",
            adaptive.get("reference_displacement"),
        )
        scalar(
            "room_adaptive_reference",
            "sampled_feature_norm",
            adaptive.get("sampled_feature_norm"),
        )
        scalar(
            "room_adaptive_reference",
            "prior_shift",
            adaptive.get("reference_prior_shift"),
        )
    room_stage = output.get("stage_rooms_logits")
    room_candidate = output.get("stage_rooms_candidate_logits")
    if isinstance(room_stage, torch.Tensor) and isinstance(
        room_candidate,
        torch.Tensor,
    ):
        pending["room_stage_candidate_delta_abs_mean"] = (
            room_candidate.detach().float() - room_stage.detach().float()
        ).abs().mean()
    adjacency = output.get("room_adjacency_logits")
    if isinstance(adjacency, torch.Tensor):
        pending["room_adjacency_probability_mean"] = (
            adjacency.detach().float().sigmoid().mean()
        )
    if not pending:
        return {}
    reference = next(iter(pending.values())).device
    host = torch.stack(
        [
            value.reshape(()).to(reference)
            for value in pending.values()
        ]
    ).cpu().tolist()
    return {
        name: float(value)
        for name, value in zip(pending, host, strict=True)
        if math.isfinite(float(value))
    }


def typed_channel_module_gradient_metrics(
    model: nn.Module,
) -> dict[str, float]:
    """Return one synchronized gradient summary for room-path modules."""

    def module_item(container, name: str):
        return getattr(container, name, None)

    stage = getattr(model, "room_instance_stage", None)
    decoder = getattr(stage, "decoder", None)
    adaptive_refiner = getattr(decoder, "adaptive_query_refiner", None)
    query_set = getattr(stage, "query_set_head", None)
    action_graph = getattr(model, "action_graph", None)
    field = getattr(model, "field", None)
    route_budget = getattr(action_graph, "route_budget", {})
    route_gate = getattr(action_graph, "route_gate", {})
    modules = {
        "room_mask_features": getattr(decoder, "mask_features", None),
        "room_mask_head": getattr(decoder, "mask_head", None),
        "room_query_embedding": getattr(decoder, "query_feature", None),
        "room_class_head": getattr(decoder, "class_head", None),
        "room_reference_head": getattr(
            adaptive_refiner,
            "reference_head",
            None,
        ),
        "room_reference_context": getattr(
            adaptive_refiner,
            "context",
            None,
        ),
        "room_query_set": getattr(stage, "query_set_head", None),
        "room_program_router": getattr(stage, "program_router", None),
        "room_graph_conditioner": getattr(
            stage,
            "room_graph_conditioner",
            None,
        ),
        "room_structure_adapter": getattr(
            stage,
            "structure_adapter",
            None,
        ),
        "room_program_class_fusion": getattr(
            getattr(decoder, "class_fusion", None),
            "program_projection",
            None,
        ),
        "room_query_route_budget_0": (
            query_set.route_budget[0]
            if query_set is not None
            and getattr(query_set, "route_budget", None)
            else None
        ),
        "room_query_route_gate_0": (
            query_set.route_gate[0]
            if query_set is not None
            and getattr(query_set, "route_gate", None)
            else None
        ),
        "field_evidence_fusion": getattr(
            field,
            "evidence_fusion",
            None,
        ),
        "action_program_route_budget": module_item(route_budget, "program"),
        "action_program_route_gate": module_item(route_gate, "program"),
    }
    names = []
    values = []
    for name, module in modules.items():
        if module is None:
            continue
        squared = [
            parameter.grad.detach().float().square().sum()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if squared:
            names.append(name)
            values.append(torch.stack(squared).sum().sqrt())
    if not values:
        return {}
    host = torch.stack(values).cpu().tolist()
    return {
        f"{name}_gradient_norm": float(value)
        for name, value in zip(names, host, strict=True)
    }


__all__ = [
    "STAGES",
    "typed_channel_module_gradient_metrics",
    "typed_channel_route_metrics",
]

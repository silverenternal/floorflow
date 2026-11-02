"""Optional room-instance and furniture stages for predicted channels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.cad_furniture_stage import CadFurnitureStage
from floorflow.model.room_instance_identity_stage import (
    RoomInstanceIdentityStage,
    owner_aware_boundary,
)


def validate_predicted_optional_stage_args(
    *,
    use_action_graph: bool,
    refresh_action_tokens: bool,
    refresh_validation_tokens: bool,
    enable_furniture_queries: bool,
    enable_instance_room_decoder: bool,
    enable_room_adjacency: bool,
    enable_room_query_set: bool,
    mask_aware_room_adjacency: bool,
    relational_room_adjacency: bool,
    count_aware_second_graph: bool,
    full_resolution_mask_refinement: bool,
) -> None:
    """Fail closed when an optional consumer has no upstream producer."""

    if enable_furniture_queries and not use_action_graph:
        raise ValueError(
            "furniture queries require the ordered CAD action graph"
        )
    if enable_instance_room_decoder and not use_action_graph:
        raise ValueError(
            "instance room decoding requires the ordered CAD action graph"
        )
    if refresh_action_tokens and not use_action_graph:
        raise ValueError(
            "action-token refresh requires the ordered CAD action graph"
        )
    if refresh_validation_tokens and not refresh_action_tokens:
        raise ValueError(
            "validation-token refresh requires action-token refresh"
        )
    if enable_room_adjacency and not enable_instance_room_decoder:
        raise ValueError(
            "room adjacency requires the instance room decoder"
        )
    if enable_room_query_set and not enable_instance_room_decoder:
        raise ValueError(
            "room query set requires the instance room decoder"
        )
    if (
        mask_aware_room_adjacency
        or relational_room_adjacency
    ) and not enable_room_adjacency:
        raise ValueError(
            "geometry-aware room adjacency requires room adjacency"
        )
    if count_aware_second_graph and not (
        enable_room_adjacency and enable_room_query_set
    ):
        raise ValueError(
            "second graph pass requires room adjacency and query set"
        )
    if full_resolution_mask_refinement and not enable_instance_room_decoder:
        raise ValueError(
            "full-resolution masks require the instance room decoder"
        )


def build_predicted_optional_stages(
    taxonomy: FloorplanTaxonomy,
    field_channels: int,
    program_channels: int,
    heads: int,
    *,
    enable_furniture_queries: bool,
    furniture_query_hidden: int,
    furniture_max_queries: int,
    furniture_min_queries: int,
    furniture_room_assignment: bool,
    furniture_semantic_router: bool,
    furniture_semantic_hidden: int,
    enable_instance_room_decoder: bool,
    room_instance_hidden: int,
    room_instance_queries: int,
    room_instance_layers: int,
    enable_program_instance_router: bool,
    enable_room_adjacency: bool,
    enable_room_query_set: bool,
    room_query_set_hidden: int,
    room_query_set_pool: int,
    room_query_set_layers: int,
    room_query_set_routing_attention: bool,
    query_count_gate: bool,
    owner_preserving_room_readout: bool,
    room_instance_adaptive_query: bool,
    room_instance_adaptive_query_residual_scale: float,
    room_instance_mask_aware_adjacency: bool,
    room_instance_relational_adjacency: bool,
    room_instance_class_count_prior: bool,
    relational_adjacency_hidden: int,
    room_instance_count_aware_graph: bool,
    count_aware_second_graph: bool,
    full_resolution_mask_refinement: bool,
) -> tuple[CadFurnitureStage | None, RoomInstanceIdentityStage | None]:
    furniture = (
        CadFurnitureStage(
            field_channels,
            len(taxonomy.furniture_ids),
            num_rooms=len(taxonomy.room_ids),
            program_channels=program_channels,
            stage_token_dim=field_channels,
            hidden_dim=furniture_query_hidden,
            max_queries=furniture_max_queries,
            min_queries=furniture_min_queries,
            enable_room_assignment=furniture_room_assignment,
            enable_semantic_router=furniture_semantic_router,
            semantic_hidden=furniture_semantic_hidden,
        )
        if enable_furniture_queries
        else None
    )
    rooms = (
        RoomInstanceIdentityStage(
            field_channels,
            len(taxonomy.room_ids),
            hidden_dim=room_instance_hidden,
            num_queries=room_instance_queries,
            layers=room_instance_layers,
            heads=heads,
            enable_program_instance_router=enable_program_instance_router,
            predict_room_adjacency=enable_room_adjacency,
            enable_query_set_head=enable_room_query_set,
            query_set_hidden=room_query_set_hidden,
            query_set_pool=room_query_set_pool,
            query_set_layers=room_query_set_layers,
            query_set_routing_attention=room_query_set_routing_attention,
            room_class_prior=(
                _room_class_prior(taxonomy)
                if room_instance_class_count_prior
                else None
            ),
            query_count_gate=query_count_gate,
            owner_preserving_room_readout=owner_preserving_room_readout,
            adaptive_query_refinement=room_instance_adaptive_query,
            adaptive_query_residual_scale=(
                room_instance_adaptive_query_residual_scale
            ),
            mask_aware_room_adjacency=room_instance_mask_aware_adjacency,
            relational_room_adjacency=(
                room_instance_relational_adjacency
            ),
            relational_adjacency_hidden=relational_adjacency_hidden,
            count_aware_room_graph=room_instance_count_aware_graph,
            count_aware_second_graph=count_aware_second_graph,
            full_resolution_mask_refinement=(
                full_resolution_mask_refinement
            ),
        )
        if enable_instance_room_decoder
        else None
    )
    return furniture, rooms


def _room_class_prior(taxonomy: FloorplanTaxonomy) -> tuple[float, ...]:
    """Return a data-provenance prior over room classes for count heads."""

    counts = taxonomy.source_counts.get("room", {})
    values = [
        float(counts.get(taxonomy.class_names[index].removeprefix("room:"), 0))
        for index in taxonomy.room_ids
    ]
    maximum = max(values, default=0.0)
    if maximum <= 0.0:
        return tuple(1.0 for _ in values)
    return tuple((value / maximum) ** 0.5 for value in values)


def opening_stage_condition(
    wall_probability: torch.Tensor,
    room_probability: torch.Tensor,
) -> torch.Tensor:
    """Build the backward-compatible opening stage condition."""

    owner_count = min(2, room_probability.shape[1])
    owner_top_two = room_probability.topk(
        owner_count,
        dim=1,
    ).values
    if owner_count == 1:
        owner_top_two = torch.cat(
            [owner_top_two, torch.zeros_like(owner_top_two)],
            dim=1,
        )
    return torch.cat(
        [
            wall_probability,
            F.max_pool2d(wall_probability, 5, stride=1, padding=2),
            1.0 - owner_top_two[:, :1],
            owner_top_two[:, 1:2],
        ],
        dim=1,
    )


def apply_opening_host_gate(
    opening_logits: torch.Tensor,
    scale: torch.Tensor,
    wall_probability: torch.Tensor,
    room_probability: torch.Tensor,
    junction_probability: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an identity-initialized host-support log prior to openings."""

    size = opening_logits.shape[-2:]
    wall = F.interpolate(
        wall_probability.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    if junction_probability is None:
        support = wall
    else:
        junction = junction_probability.float()
        if junction.ndim == 3:
            junction = junction[:, None]
        junction = F.interpolate(
            junction,
            size=size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        boundary = owner_aware_boundary(room_probability)
        host_evidence = torch.maximum(junction, boundary)
        support = wall * (0.5 + 0.5 * host_evidence)
    log_prior = support.clamp(1.0e-4, 1.0).log()
    return (
        opening_logits + torch.tanh(scale) * log_prior,
        support,
    )


def refresh_validation_action(action_graph, action_inputs, furniture_logits):
    """Bind final furniture evidence before the validation CAD stage."""

    furniture_probability = furniture_logits.sigmoid()
    action_inputs["furniture"] = torch.cat(
        [
            furniture_probability.amax(dim=1, keepdim=True),
            furniture_logits.softmax(dim=1),
        ],
        dim=1,
    )
    return action_graph(action_inputs)


def resolve_field_features(
    model,
    controls: dict[str, torch.Tensor],
    footprint: torch.Tensor,
    cached_features: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Use cached frozen field evidence or run the field encoder."""

    if cached_features is None:
        return model.field.encode_details(controls, footprint)
    missing = [
        name
        for name in ("field_features", "source_tokens")
        if name not in cached_features
    ]
    if missing:
        raise ValueError(f"cached field features missing: {missing}")
    features = cached_features["field_features"]
    evidence = cached_features.get("evidence")
    source_tokens = cached_features["source_tokens"]
    if model.control_evidence_context and evidence is None:
        raise ValueError(
            "cached field features require evidence for control_evidence_context"
        )
    batch = footprint.shape[0]
    if features.ndim != 4 or features.shape[0] != batch:
        raise ValueError("cached field_features must be [B,C,H,W]")
    if source_tokens.ndim != 3 or source_tokens.shape[0] != batch:
        raise ValueError("cached source_tokens must be [B,S,D]")
    if evidence is not None and (
        evidence.ndim != 2 or evidence.shape[0] != batch
    ):
        raise ValueError("cached evidence must be [B,D]")
    target_device = footprint.device
    if any(
        value.device != target_device
        for value in (features, source_tokens, evidence)
        if value is not None
    ):
        raise ValueError("cached field features must share the footprint device")
    return features, evidence, source_tokens


__all__ = [
    "apply_opening_host_gate",
    "build_predicted_optional_stages",
    "opening_stage_condition",
    "refresh_validation_action",
    "resolve_field_features",
    "validate_predicted_optional_stage_args",
]

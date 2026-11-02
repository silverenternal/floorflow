"""All-channel predictor over real predicted-control fields."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.channel_reliability import reliability_weighted_residual
from floorflow.heads.masked_channel_distribution import (
    MaskedChannelDistributionHead,
    TypedChannelDistributionHead,
)
from floorflow.model.cad_action_graph import CadActionGraph
from floorflow.model.cad_stage_heads import build_cad_stage_heads
from floorflow.model.cad_stage_readout import apply_cad_stage
from floorflow.model.predicted_action_context import predicted_action_inputs
from floorflow.model.predicted_control_prior import predicted_control_prior
from floorflow.model.predicted_field_encoder import PredictedControlFieldEncoder
from floorflow.model.predicted_furniture_runtime import attach_furniture_queries
from floorflow.model.predicted_optional_stages import (
    apply_opening_host_gate,
    build_predicted_optional_stages,
    opening_stage_condition,
    refresh_validation_action,
    resolve_field_features,
    validate_predicted_optional_stage_args,
)
from floorflow.model.predicted_room_runtime import (
    apply_action_residual_gates,
    attach_cad_layers,
    attach_room_instance,
    routed_program_inputs,
    taxonomy_class_indices,
)


class PredictedChannelModel(nn.Module):
    """Predict all semantic channels, optionally with typed channel links."""

    def __init__(
        self,
        taxonomy: FloorplanTaxonomy,
        *,
        control_channels: tuple[int, int, int, int] = (24, 38, 52, 28),
        program_channels: int = 24,
        grid: int = 64,
        field_channels: int = 48,
        head_channels: int = 96,
        heads: int = 4,
        layers: int = 2,
        attention_grid: int = 16,
        full_resolution_output: bool = True,
        typed_channel_links: bool = True,
        use_action_graph: bool = True,
        preserve_room_stage: bool = True,
        enable_furniture_queries: bool = False,
        furniture_query_hidden: int = 96,
        furniture_max_queries: int = 24,
        furniture_min_queries: int = 4,
        furniture_detach_backbone: bool = False,
        furniture_room_assignment: bool = False,
        furniture_semantic_router: bool = False,
        furniture_semantic_hidden: int = 64,
        enable_instance_room_decoder: bool = False,
        room_instance_hidden: int = 96,
        room_instance_queries: int = 32,
        room_instance_layers: int = 2,
        enable_program_instance_router: bool = False,
        program_room_action_fusion: bool = False,
        enable_room_adjacency: bool = False,
        enable_room_query_set: bool = False,
        room_query_set_hidden: int = 64,
        room_query_set_pool: int = 48,
        room_query_set_layers: int = 2,
        room_query_set_routing_attention: bool = True,
        query_count_gate: bool = True,
        owner_preserving_room_readout: bool = True,
        room_instance_adaptive_query: bool = False,
        room_instance_adaptive_query_residual_scale: float = 0.0,
        room_instance_mask_aware_adjacency: bool = False,
        room_instance_relational_adjacency: bool = False,
        room_instance_class_count_prior: bool = False,
        relational_adjacency_hidden: int = 128,
        room_instance_count_aware_graph: bool = False,
        count_aware_second_graph: bool = False,
        full_resolution_mask_refinement: bool = False,
        typed_opening_host_gate: bool = False,
        control_evidence_context: bool = False,
        action_context_conditioning: bool = False,
        typed_source_spatial_router: bool = False,
        dynamic_channel_links: bool = False,
        dynamic_channel_queries: bool = False,
        source_channel_attention: bool = False,
        dynamic_cross_attention: bool = False,
        channel_group_adapters: bool = False,
        cross_attention_lora_rank: int = 0,
        spatial_conv_adapter: bool = False,
        spatial_adapter_bottleneck: int = 16,
        source_hyper_hidden: int = 64,
        dynamic_stage_heads: bool = False,
        stage_head_hidden: int = 64,
        dynamic_stage_links: bool = False,
        action_route_attention: bool = False,
        action_route_router_count: int = 8,
        action_route_top_k: int = 4,
        differentiable_route_budget: bool = False,
        refresh_action_tokens: bool = False,
        refresh_validation_tokens: bool = False,
        stage_route_attention: bool = False,
        stage_route_grid: int = 8,
        stage_route_router_count: int = 12,
        stage_route_top_k: int = 6,
        stage_route_heads: int = 4,
    ) -> None:
        super().__init__()
        self.program_room_action_fusion = bool(program_room_action_fusion)
        self.differentiable_route_budget = bool(
            differentiable_route_budget
        )
        validate_predicted_optional_stage_args(
            use_action_graph=use_action_graph,
            refresh_action_tokens=refresh_action_tokens,
            refresh_validation_tokens=refresh_validation_tokens,
            enable_furniture_queries=enable_furniture_queries,
            enable_instance_room_decoder=enable_instance_room_decoder,
            enable_room_adjacency=enable_room_adjacency,
            enable_room_query_set=enable_room_query_set,
            mask_aware_room_adjacency=room_instance_mask_aware_adjacency,
            relational_room_adjacency=room_instance_relational_adjacency,
            count_aware_second_graph=count_aware_second_graph,
            full_resolution_mask_refinement=full_resolution_mask_refinement,
        )
        self.taxonomy = taxonomy
        self.field = PredictedControlFieldEncoder(
            control_channels,
            program_channels,
            grid=grid,
            hidden_channels=field_channels,
            evidence_context=control_evidence_context,
            typed_source_spatial_router=typed_source_spatial_router,
        )
        head_type = (
            TypedChannelDistributionHead
            if typed_channel_links
            else MaskedChannelDistributionHead
        )
        self.head = (
            head_type(
                field_channels,
                taxonomy,
                hidden_channels=head_channels,
                heads=heads,
                layers=layers,
                grid=attention_grid,
                full_resolution_output=full_resolution_output,
                context_dim=(
                    field_channels
                    if control_evidence_context or action_context_conditioning
                    else 0
                ),
                dynamic_channel_links=dynamic_channel_links,
                dynamic_channel_queries=dynamic_channel_queries,
                source_attention=source_channel_attention,
                dynamic_cross_attention=dynamic_cross_attention,
                channel_group_adapters=channel_group_adapters,
                cross_attention_lora_rank=cross_attention_lora_rank,
                spatial_conv_adapter=spatial_conv_adapter,
                spatial_adapter_bottleneck=spatial_adapter_bottleneck,
                source_count=6,
                source_hyper_hidden=source_hyper_hidden,
            )
            if typed_channel_links
            else head_type(
                field_channels,
                taxonomy.num_classes,
                hidden_channels=head_channels,
                heads=heads,
                layers=layers,
                grid=attention_grid,
                full_resolution_output=full_resolution_output,
                context_dim=(
                    field_channels
                    if control_evidence_context or action_context_conditioning
                    else 0
                ),
                dynamic_channel_queries=dynamic_channel_queries,
                source_attention=source_channel_attention,
                dynamic_cross_attention=dynamic_cross_attention,
                cross_attention_lora_rank=cross_attention_lora_rank,
                spatial_conv_adapter=spatial_conv_adapter,
                spatial_adapter_bottleneck=spatial_adapter_bottleneck,
                source_count=6,
                source_hyper_hidden=source_hyper_hidden,
            )
        )
        self.control_evidence_context = bool(control_evidence_context)
        self.action_context_conditioning = bool(
            action_context_conditioning
        )
        self.typed_source_spatial_router = bool(
            typed_source_spatial_router
        )
        self.refresh_action_tokens = bool(refresh_action_tokens)
        self.refresh_validation_tokens = bool(refresh_validation_tokens)
        self.dynamic_channel_links = bool(dynamic_channel_links)
        self.dynamic_channel_queries = bool(dynamic_channel_queries)
        self.source_channel_attention = bool(source_channel_attention)
        self.dynamic_cross_attention = bool(dynamic_cross_attention)
        self.typed_channel_links = bool(typed_channel_links)
        self.preserve_room_stage = bool(preserve_room_stage)
        self.furniture_detach_backbone = bool(furniture_detach_backbone)
        self.head_scale = nn.Parameter(torch.tensor(0.05))
        self.opening_host_gate_scale = nn.Parameter(torch.tensor(1.0e-3)) if typed_opening_host_gate else None
        self.action_context_projection = (
            nn.Linear(field_channels, field_channels)
            if action_context_conditioning
            else None
        )
        if self.action_context_projection is not None:
            nn.init.normal_(self.action_context_projection.weight, std=0.01)
            nn.init.zeros_(self.action_context_projection.bias)
        self.action_graph = (
            CadActionGraph(
                {
                    "envelope": 1,
                    "program": program_channels,
                    "rooms": len(taxonomy.room_ids),
                    "walls": 1,
                    "junctions": 1,
                    "openings": 1 + len(taxonomy.opening_ids),
                    "furniture": len(taxonomy.furniture_ids) + 1,
                    "validation": control_channels[3],
                },
                hidden_dim=field_channels,
                heads=heads,
                dynamic_stage_links=dynamic_stage_links,
                route_attention=action_route_attention,
                route_router_count=action_route_router_count,
                route_top_k=action_route_top_k,
                differentiable_route_budget=(
                    differentiable_route_budget
                ),
            )
            if use_action_graph
            else None
        )
        self.dynamic_stage_heads = bool(dynamic_stage_heads)
        self.stage_route_attention = bool(stage_route_attention)
        self.stage_hidden = int(stage_head_hidden)
        (
            self.stage_refiners,
            self.stage_trunks,
            self.stage_film,
            self.stage_heads,
        ) = build_cad_stage_heads(
            taxonomy,
            field_channels,
            self.stage_hidden,
            dynamic=self.dynamic_stage_heads,
            route_attention=self.stage_route_attention,
            route_grid=stage_route_grid,
            route_router_count=stage_route_router_count,
            route_top_k=stage_route_top_k,
            route_heads=stage_route_heads,
        )
        self.stage_scales = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(0.1))
                for name in ("rooms", "walls", "openings", "furniture")
            }
        )
        self.furniture_stage, self.room_instance_stage = (
            build_predicted_optional_stages(
                taxonomy,
                field_channels,
                program_channels,
                heads,
                enable_furniture_queries=enable_furniture_queries,
                furniture_query_hidden=furniture_query_hidden,
                furniture_max_queries=furniture_max_queries,
                furniture_min_queries=furniture_min_queries,
                furniture_room_assignment=furniture_room_assignment,
                furniture_semantic_router=furniture_semantic_router,
                furniture_semantic_hidden=furniture_semantic_hidden,
                enable_instance_room_decoder=enable_instance_room_decoder,
                room_instance_hidden=room_instance_hidden,
                room_instance_queries=room_instance_queries,
                room_instance_layers=room_instance_layers,
                enable_program_instance_router=enable_program_instance_router,
                enable_room_adjacency=enable_room_adjacency,
                enable_room_query_set=enable_room_query_set,
                room_query_set_hidden=room_query_set_hidden,
                room_query_set_pool=room_query_set_pool,
                room_query_set_layers=room_query_set_layers,
                room_query_set_routing_attention=room_query_set_routing_attention,
                query_count_gate=query_count_gate,
                owner_preserving_room_readout=owner_preserving_room_readout,
                room_instance_adaptive_query=room_instance_adaptive_query,
                room_instance_adaptive_query_residual_scale=room_instance_adaptive_query_residual_scale,
                room_instance_mask_aware_adjacency=room_instance_mask_aware_adjacency,
                room_instance_relational_adjacency=room_instance_relational_adjacency,
                room_instance_class_count_prior=room_instance_class_count_prior,
                relational_adjacency_hidden=relational_adjacency_hidden,
                room_instance_count_aware_graph=room_instance_count_aware_graph,
                count_aware_second_graph=count_aware_second_graph,
                full_resolution_mask_refinement=full_resolution_mask_refinement,
            )
        )

    def forward(
        self,
        controls: dict[str, torch.Tensor],
        footprint: torch.Tensor,
        cached_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if cached_features is not None:
            self.field.last_source_routing = None
        features, evidence, source_tokens = resolve_field_features(
            self, controls, footprint, cached_features
        )
        action_inputs = (
            predicted_action_inputs(self.taxonomy, controls, footprint)
            if self.action_graph is not None
            else None
        )
        action = (
            self.action_graph(action_inputs)
            if self.action_graph is not None and action_inputs is not None
            else None
        )
        if (
            self.action_context_conditioning
            and self.action_context_projection is not None
            and action is not None
        ):
            action_context = torch.stack(
                list(action.tokens.values()),
                dim=1,
            ).mean(dim=1)
            projected = self.action_context_projection(action_context)
            evidence = (
                projected
                if evidence is None
                else evidence + projected
            )
        output = self.head(
            features,
            context=evidence,
            source_tokens=source_tokens,
        )
        prior = predicted_control_prior(
            self.taxonomy,
            controls,
            self.field.grid,
        )
        residual = output["logits"]
        residual = (
            reliability_weighted_residual(residual, output["reliability"])
            if self.typed_source_spatial_router
            else residual
        )
        room_ids, opening_ids, furniture_ids = taxonomy_class_indices(
            self.taxonomy,
            residual.device,
        )
        residual = apply_action_residual_gates(
            residual,
            action,
            self.taxonomy,
        )
        combined = prior + self.head_scale.to(residual.dtype) * residual
        if self.field.last_source_routing is not None:
            output["field_source_routing"] = self.field.last_source_routing
        stage_tokens = action.tokens if action is not None else None
        stage_gates = action.gates if action is not None else None
        structure_features = features
        if stage_tokens is not None:
            wall_id = self.taxonomy.class_id("wall")
            wall_ids = torch.tensor([wall_id], device=combined.device)

            def apply_stage(
                name: str,
                class_ids: torch.Tensor,
                stage_features: torch.Tensor,
                condition: torch.Tensor | None = None,
            ) -> torch.Tensor:
                result = apply_cad_stage(
                    name,
                    class_ids,
                    stage_features,
                    stage_tokens[name],
                    stage_gates[name],
                    self.stage_scales[name],
                    prior,
                    dynamic_refiner=(
                        self.stage_refiners[name]
                        if self.dynamic_stage_heads
                        else None
                    ),
                    legacy_trunk=(
                        self.stage_trunks[name]
                        if not self.dynamic_stage_heads
                        else None
                    ),
                    legacy_film=(
                        self.stage_film[name]
                        if not self.dynamic_stage_heads
                        else None
                    ),
                    legacy_head=(
                        self.stage_heads[name]
                        if not self.dynamic_stage_heads
                        else None
                    ),
                    preserve_room_stage=self.preserve_room_stage,
                    condition=condition,
                    base_logits=combined[:, class_ids],
                )
                combined[:, class_ids] = result["logits"]
                output[f"stage_{name}_logits"] = result["logits"]
                output[f"stage_{name}_candidate_logits"] = result[
                    "candidate_logits"
                ]
                if result["spatial_gate"] is not None:
                    output[f"stage_{name}_spatial_gate"] = result[
                        "spatial_gate"
                    ]
                if result.get("route_diagnostics") is not None:
                    output[f"stage_{name}_route_diagnostics"] = result[
                        "route_diagnostics"
                    ]
                return result["logits"]

            room_stage_logits = apply_stage(
                "rooms",
                room_ids,
                features,
            )
            if (
                self.refresh_action_tokens
                and self.action_graph is not None
                and action_inputs is not None
            ):
                action_inputs["rooms"] = room_stage_logits
                action = self.action_graph(action_inputs)
                stage_tokens = action.tokens
                stage_gates = action.gates
            if self.room_instance_stage is not None:
                room_stage_candidate_logits = output[
                    "stage_rooms_candidate_logits"
                ]
                program_controls, program_route_token = routed_program_inputs(
                    controls,
                    action,
                    fuse_room_action=self.program_room_action_fusion,
                )
                room_instance = attach_room_instance(
                    output,
                    self.room_instance_stage,
                    features,
                    room_stage_logits,
                    action.tokens["rooms"],
                    program_controls,
                    program_action_token=program_route_token,
                    room_stage_candidate_logits=room_stage_candidate_logits,
                    full_resolution_room_logits=controls["room_logits"],
                )
                combined[:, room_ids] = room_instance[
                    "room_instance_logits"
                ]
                structure_features = room_instance["structure_features"]
                if (
                    self.refresh_action_tokens
                    and self.action_graph is not None
                    and action_inputs is not None
                ):
                    action_inputs["rooms"] = room_instance[
                        "room_instance_logits"
                    ]
                    action = self.action_graph(action_inputs)
                    stage_tokens = action.tokens
                    stage_gates = action.gates
            room_probability = combined[:, room_ids].softmax(dim=1)
            wall_probability = combined[:, wall_ids].sigmoid()
            wall_stage_logits = apply_stage(
                "walls",
                wall_ids,
                structure_features,
                condition=room_probability,
            )
            if (
                self.refresh_action_tokens
                and self.action_graph is not None
                and action_inputs is not None
            ):
                action_inputs["walls"] = wall_stage_logits
                action = self.action_graph(action_inputs)
                stage_tokens = action.tokens
                stage_gates = action.gates
                wall_probability = wall_stage_logits.sigmoid()
            opening_condition = opening_stage_condition(
                wall_probability,
                room_probability,
            )
            opening_stage_logits = apply_stage(
                "openings",
                opening_ids,
                structure_features,
                condition=opening_condition,
            )
            if self.opening_host_gate_scale is not None:
                opening_stage_logits, host_support = apply_opening_host_gate(
                    opening_stage_logits,
                    self.opening_host_gate_scale,
                    wall_probability,
                    room_probability,
                    controls.get("junction_probability"),
                )
                combined[:, opening_ids] = opening_stage_logits
                output["stage_openings_logits"] = opening_stage_logits
                output["opening_host_support"] = host_support
                output["opening_host_gate_scale"] = (
                    self.opening_host_gate_scale.detach()
                )
            if (
                self.refresh_action_tokens
                and self.action_graph is not None
                and action_inputs is not None
            ):
                opening_probability = opening_stage_logits.sigmoid()
                action_inputs["openings"] = torch.cat(
                    [
                        opening_probability.amax(dim=1, keepdim=True),
                        opening_stage_logits.softmax(dim=1),
                    ],
                    dim=1,
                )
                action = self.action_graph(action_inputs)
                stage_tokens = action.tokens
                stage_gates = action.gates
            furniture_stage_logits = apply_stage(
                "furniture",
                furniture_ids,
                structure_features,
                condition=room_probability,
            )
            if self.refresh_validation_tokens:
                action = refresh_validation_action(
                    self.action_graph,
                    action_inputs,
                    furniture_stage_logits,
                )
                output["action_validation_refreshed"] = combined.new_tensor(1.0)
            if self.refresh_action_tokens:
                refreshes = 4.0 if self.refresh_validation_tokens else 3.0
                output["action_refresh_count"] = combined.new_tensor(refreshes)
        output["prior_logits"] = prior
        output["residual_logits"] = output["logits"]
        output["logits"] = combined
        output["room_logits"] = combined[:, room_ids]
        attach_cad_layers(
            output,
            combined,
            self.taxonomy,
            room_ids,
            opening_ids,
            furniture_ids,
        )
        output["field_features"] = features
        if action is not None:
            output["action_tokens"] = action.tokens
            output["action_stage_tokens"] = torch.stack(
                list(action.tokens.values()),
                dim=1,
            )
            output["action_gates"] = action.gates
            output["action_route_diagnostics"] = (
                action.route_diagnostics
            )
            output["validation_token"] = action.tokens["validation"]
            if self.furniture_stage is not None:
                attach_furniture_queries(
                    output,
                    self.furniture_stage,
                    structure_features,
                    controls,
                    footprint,
                    action.tokens["furniture"],
                    combined[:, room_ids],
                    self.furniture_detach_backbone,
                )
        return output


__all__ = ["PredictedChannelModel", "PredictedControlFieldEncoder"]

"""Coupled L0-L3 flow states for typed chip graph conditioning."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

TRINITY_SOURCE = Path(__file__).resolve().parents[3] / "third_party/Trinity/src"
if str(TRINITY_SOURCE) not in sys.path:
    sys.path.insert(0, str(TRINITY_SOURCE))

from trinity.conditioning.graph_pe import SpectralDrawPE
from trinity.models.backbone import SetTransformerDenoiser
from trinity.models.presets import DenoiserArchConfig

from floorflow.chip_flow import hierarchy_coordinates
from floorflow.chip_flow.backbone_lora import install_backbone_lora
from floorflow.chip_flow.boundary_anchor_coupler import BoundaryAnchorCoupler
from floorflow.chip_flow.chip_hierarchy_conditioning import ChipHierarchyConditioners
from floorflow.chip_flow.global_bbox_consensus import GlobalBBoxConsensus
from floorflow.chip_flow.group_moment_coupler import NativeGroupMomentModule, apply_native_group_moment
from floorflow.chip_flow.hierarchy_adapters import (
    apply_hierarchy_input_adapters,
    apply_hierarchy_output_adapters,
    install_optional_hierarchy_adapters,
)
from floorflow.chip_flow.hierarchy_context import prepare_hierarchy_context
from floorflow.chip_flow.hierarchy_state import HierarchyContext, HierarchyPrediction, HierarchyState, StateEdgeContext
from floorflow.chip_flow.hierarchy_structure import structural_membership
from floorflow.chip_flow.hierarchy_targets import build_hierarchy_targets
from floorflow.chip_flow.hierarchy_token_bridge import HierarchyTokenBridgeInput, TokenBridgeBackbone
from floorflow.chip_flow.observed_hierarchy import ObservedHierarchyCoupler
from floorflow.chip_flow.outline_conditions import outline_condition_from_boxes, outline_condition_from_hw
from floorflow.chip_flow.projector_residual_head import ProjectorResidualHead
from floorflow.chip_flow.sparse_layers import StateEdgeRelay, masked_token_mean
from floorflow.chip_flow.state_conditioned_edge_bias import StateConditionedEdgeBias, condition_state_edges
from floorflow.chip_flow.structural_relations import StructuralRelationHead
from floorflow.chip_flow.structured_heads import StructuredHeadSuite
from floorflow.chip_flow.typed_graph_encoder import TypedGraphEncoder


class HierarchicalTokenFlowModel(nn.Module):
    """Joint L0-L3 state model with the established Trinity block backbone."""

    def __init__(
        self,
        hidden: int = 192,
        heads: int = 6,
        depth: int = 12,
        head_dim: int | None = None,
        graph_pe_k: int = 2,
        typed_layers: int = 2,
        typed_mode: str = "typed",
        relation_specific: bool = False,
        incidence_normalized_relations: bool = False,
        structural_relations: bool = False,
        complexity_conditioning: bool = False,
        coarse_commit_fraction: float | None = None,
        coarse_anchor_tolerance: float = 0.05,
        coarse_anchor_strength: float = 0.0,
        coarse_anchor_max_step: float = 0.1,
        use_hierarchy: bool = True,
        use_netlist: bool = True,
        netlist_edge_gate: bool = False,
        projector: bool = False,
        projector_iterations: int = 4,
        projector_boundary_axes: bool = False,
        projector_boundary_lock_corners: bool = True,
        projector_sparse_pairs: bool = True,
        projector_strategy: str = "legacy",
        projector_grid_size: int = 16,
        hierarchy_global_weight: float = 0.1,
        hierarchy_structural_weight: float = 0.1,
        hierarchy_port_weight: float = 0.1,
        hierarchy_bbox_weight: float = 0.0,
        hierarchy_bbox_decay_steps: int = 0,
        outline_conditioning: bool = False,
        dynamic_graph_relay: bool = False,
        dynamic_graph_input: bool = False,
        order_conditioning: bool = False,
        coarse_to_fine_conditioning: bool = False,
        port_consensus_conditioning: bool = False,
        port_consensus_output: bool = False,
        boundary_anchor_conditioning: bool = False,
        spatial_occupancy_conditioning: bool = False,
        spatial_route_field: bool = False,
        symmetric_b2b_messages: bool = False,
        complexity_slot_conditioning: bool = False,
        degree_expert_conditioning: bool = False,
        hierarchy_token_bridge: bool = False,
        hierarchy_token_bridge_mode: str = "legacy",
        global_bbox_consensus: bool = False,
        self_conditioning: bool = False,
        backbone_lora_rank: int = 0,
        backbone_lora_alpha: float | None = None,
        projector_residual_head: bool = False,
        state_conditioned_edge_bias: bool = False,
        structured_heads: bool = False,
        structured_hierarchy_coupling: bool = True,
        structured_factorized_block: bool = True,
        structured_factorized_shared_trunk: bool = False,
        structured_rho_trust_radius: float = 0.0,
        structured_dual_ports: bool = True,
        structured_legalizer_response: bool = True,
        structured_legalizer_apply: bool = False,
        structured_legalizer_strength: float = 1.0,
        structured_legalizer_max_displacement: float = 0.1,
        global_spec_conditioning: bool = False,
        node_semantic_dim: int = 0,
        port_semantic_dim: int = 0,
        netlist_hyperedge_conditioning: bool = False,
        membership_hierarchy_conditioning: bool = False,
        standalone_legalizer_response: bool = False,
        observed_hierarchy_coupling: bool = False,
        hierarchy_coordinate_parameterization: bool = False,
        hierarchy_coordinate_mode: str = "envelope_cluster",
        group_moment_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.heads = int(heads)
        self.head_dim = int(head_dim if head_dim is not None else hidden // 3)
        self.graph_pe_k = int(graph_pe_k)
        self.typed_mode = typed_mode
        self.coarse_commit_fraction = coarse_commit_fraction
        if coarse_commit_fraction is not None and not 0.0 <= coarse_commit_fraction <= 1.0:
            raise ValueError("coarse commit fraction must be in [0, 1]")
        if coarse_anchor_tolerance < 0.0:
            raise ValueError("coarse anchor tolerance must be non-negative")
        if not 0.0 <= coarse_anchor_strength <= 1.0:
            raise ValueError("coarse anchor strength must be in [0, 1]")
        if coarse_anchor_max_step < 0.0:
            raise ValueError("coarse anchor max step must be non-negative")
        self.coarse_anchor_tolerance = float(coarse_anchor_tolerance)
        self.coarse_anchor_strength = float(coarse_anchor_strength)
        self.coarse_anchor_max_step = float(coarse_anchor_max_step)
        self.use_hierarchy = bool(use_hierarchy)
        self.use_netlist = bool(use_netlist)
        self.netlist_edge_gate = bool(netlist_edge_gate)
        self.projector = bool(projector)
        self.projector_iterations = int(projector_iterations)
        self.projector_boundary_axes = bool(projector_boundary_axes)
        self.projector_boundary_lock_corners = bool(projector_boundary_lock_corners)
        self.projector_sparse_pairs = bool(projector_sparse_pairs)
        self.projector_strategy = str(projector_strategy)
        self.projector_grid_size = int(projector_grid_size)
        self.hierarchy_global_weight = float(hierarchy_global_weight)
        self.hierarchy_structural_weight = float(hierarchy_structural_weight)
        self.hierarchy_port_weight = float(hierarchy_port_weight)
        self.hierarchy_bbox_weight = float(hierarchy_bbox_weight)
        self.hierarchy_bbox_decay_steps = int(hierarchy_bbox_decay_steps)
        self.outline_conditioning = bool(outline_conditioning)
        self.dynamic_graph_relay = bool(dynamic_graph_relay)
        self.dynamic_graph_input = bool(dynamic_graph_input)
        self.order_conditioning = bool(order_conditioning)
        self.coarse_to_fine_conditioning = bool(coarse_to_fine_conditioning)
        self.port_consensus_conditioning = bool(port_consensus_conditioning)
        self.port_consensus_output = bool(port_consensus_output)
        self.boundary_anchor_conditioning = bool(boundary_anchor_conditioning)
        self.observed_hierarchy_coupling = bool(observed_hierarchy_coupling)
        self.observed_hierarchy_coupler = ObservedHierarchyCoupler(hidden) if (
            self.observed_hierarchy_coupling
        ) else None
        self.spatial_occupancy_conditioning = bool(spatial_occupancy_conditioning)
        self.spatial_route_field = bool(spatial_route_field)
        self.symmetric_b2b_messages = bool(symmetric_b2b_messages)
        self.complexity_slot_conditioning = bool(complexity_slot_conditioning)
        self.hierarchy_token_bridge = bool(hierarchy_token_bridge)
        self.hierarchy_coordinate_parameterization = bool(
            hierarchy_coordinate_parameterization
        )
        if group_moment_conditioning and not structural_relations:
            raise ValueError("group moments require structural relations")
        self.hierarchy_coordinate_mode = str(hierarchy_coordinate_mode)
        self.coordinate_gate = hierarchy_coordinates.HierarchyBlockCoordinateGate(hidden) if self.hierarchy_coordinate_parameterization else None
        self.global_bbox_consensus = GlobalBBoxConsensus() if global_bbox_consensus else None
        self.self_conditioning = bool(self_conditioning)
        self.backbone_lora_rank = int(backbone_lora_rank)
        self.projector_residual_head = ProjectorResidualHead(hidden) if projector_residual_head else None
        self.state_conditioned_edge_bias = StateConditionedEdgeBias() if state_conditioned_edge_bias else None
        self.relation_head = (
            StructuralRelationHead(hidden) if structural_relations else None
        )
        self.group_moment_coupler = NativeGroupMomentModule(hidden) if group_moment_conditioning else None
        self.structured_heads = (
            StructuredHeadSuite(
                hidden,
                hierarchy_coupling=structured_hierarchy_coupling,
                factorized_block=structured_factorized_block,
                factorized_shared_trunk=structured_factorized_shared_trunk,
                factorized_rho_trust_radius=structured_rho_trust_radius,
                dual_semantic_ports=structured_dual_ports,
                legalizer_response=structured_legalizer_response,
            )
            if structured_heads
            else None
        )
        if not 0.0 <= structured_legalizer_strength <= 1.0:
            raise ValueError("structured legalizer strength must be in [0, 1]")
        self.structured_legalizer_apply = bool(structured_legalizer_apply)
        self.structured_legalizer_strength = float(structured_legalizer_strength)
        if structured_legalizer_max_displacement < 0.0:
            raise ValueError("structured legalizer trust radius must be non-negative")
        self.structured_legalizer_max_displacement = float(structured_legalizer_max_displacement)
        self.output_mode = "x0"
        self.typed = TypedGraphEncoder(
            hidden=hidden,
            heads=heads,
            graph_layers=typed_layers,
            typed_mode=typed_mode,
            edge_gate=self.netlist_edge_gate,
            symmetric_b2b_messages=self.symmetric_b2b_messages,
            degree_expert_conditioning=degree_expert_conditioning,
            relation_specific=relation_specific,
            incidence_normalized_relations=incidence_normalized_relations,
            complexity_conditioning=complexity_conditioning,
            global_spec_conditioning=global_spec_conditioning,
            node_semantic_dim=node_semantic_dim,
            port_semantic_dim=port_semantic_dim,
        )
        self.conditioners = ChipHierarchyConditioners(
            hidden,
            structured_heads=self.structured_heads,
            membership_hierarchy=membership_hierarchy_conditioning,
            netlist_hyperedge=netlist_hyperedge_conditioning,
            legalizer_response=standalone_legalizer_response,
        )
        backbone_config = DenoiserArchConfig(
            latent_dim=3,
            feature_dim=19,
            graph_pe_dim=self.graph_pe_k,
            hidden=hidden,
            depth=self.depth,
            heads=3,
            head_dim=self.head_dim,
            norm="rmsnorm",
            mlp="swiglu",
            mlp_ratio=4.0,
            attn="sdpa_graph",
            qk_norm=True,
            graph_bias=True,
            graph_mix={"name": "mp", "normalize": "row"},
            time_cond="token",
            output_kind="x0",
            post_norm=False,
            time_scale=1000.0,
        )
        self.backbone = (
            TokenBridgeBackbone(
                backbone_config,
                bridge_heads=heads,
                bridge_mode=hierarchy_token_bridge_mode,
            )
            if self.hierarchy_token_bridge
            else SetTransformerDenoiser(backbone_config)
        )
        if self.backbone_lora_rank > 0:
            install_backbone_lora(
                self.backbone,
                rank=self.backbone_lora_rank,
                alpha=float(
                    backbone_lora_alpha
                    if backbone_lora_alpha is not None
                    else self.backbone_lora_rank
                ),
            )
        self.graph_pe = SpectralDrawPE(k=self.graph_pe_k, normalize=None)
        self.time = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.hierarchy_summary = nn.Sequential(
            nn.Linear(9, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.hierarchy_to_latent = nn.Linear(hidden + hidden, 3)
        nn.init.zeros_(self.hierarchy_to_latent.weight)
        nn.init.zeros_(self.hierarchy_to_latent.bias)
        self.history_to_latent = None
        if self.self_conditioning:
            width = max(8, hidden // 24)
            self.history_to_latent = nn.Sequential(
                nn.Linear(3, width),
                nn.SiLU(),
                nn.Linear(width, 3),
            )
            nn.init.zeros_(self.history_to_latent[-1].weight)
            nn.init.zeros_(self.history_to_latent[-1].bias)
        self.global_head = nn.Sequential(
            nn.Linear(hidden * 2 + 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),
        )
        self.structural_head = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        self.port_head = nn.Sequential(
            nn.Linear(hidden * 2 + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        self.outline_to_hidden = None
        if self.outline_conditioning:
            with torch.random.fork_rng(devices=[]):
                self.outline_to_hidden = nn.Linear(4, hidden)
            nn.init.zeros_(self.outline_to_hidden.weight)
            nn.init.zeros_(self.outline_to_hidden.bias)
        self.dynamic_relay = (
            StateEdgeRelay(state_dim=3)
            if self.dynamic_graph_relay or self.dynamic_graph_input
            else None
        )
        self.dynamic_to_latent = (
            nn.Linear(3, 3) if self.dynamic_graph_input else None
        )
        if self.dynamic_to_latent is not None:
            # The relay gate remains zero, so this projection preserves the
            # parent output while keeping the relay gate gradient reachable.
            nn.init.normal_(self.dynamic_to_latent.weight, std=0.02)
            nn.init.zeros_(self.dynamic_to_latent.bias)
        self.boundary_anchor = BoundaryAnchorCoupler(hidden) if self.boundary_anchor_conditioning else None
        install_optional_hierarchy_adapters(self, hidden)

    def prepare_context(self, batch: dict) -> HierarchyContext:
        return prepare_hierarchy_context(self, batch)

    def forward(
        self,
        state: HierarchyState,
        time: torch.Tensor,
        context: HierarchyContext,
        *,
        apply_legalizer_action: bool = False,
    ) -> HierarchyPrediction:
        time_token = self.time(time[:, None].float())
        typed = context.typed
        structural_mean = masked_token_mean(
            state.structural_state,
            typed.structural_mask,
        )
        if self.typed_mode == "no_structural":
            structural_mean = torch.zeros_like(structural_mean)
        port_mean = masked_token_mean(state.port_state, typed.port_mask)
        outline_token = None
        if self.outline_to_hidden is not None:
            if context.outline_state is None:
                raise ValueError("outline conditioning requires outline state")
            outline_token = self.outline_to_hidden(context.outline_state)
        global_token = typed.global_token
        structural_tokens = typed.structural_tokens
        port_tokens = typed.port_tokens
        if self.observed_hierarchy_coupler is not None:
            if (
                context.block_areas is None
                or context.coarse_membership is None
            ):
                raise ValueError(
                    "observed hierarchy coupling requires areas and membership"
                )
            global_token, structural_tokens, port_tokens, _ = (
                self.observed_hierarchy_coupler(
                    state.block_state, global_token, structural_tokens,
                    port_tokens, context.block_areas, context.coarse_membership,
                    typed.structural_mask, typed.port_block_incidence,
                    typed.port_mask,
                )
            )
        if outline_token is not None:
            global_token = global_token + outline_token[:, None]
        global_prediction = self.global_head(
            torch.cat(
                [state.global_state, global_token[:, 0], time_token],
                dim=-1,
            )
        )
        relation_features = None
        if self.relation_head is not None:
            if context.coarse_membership is None or context.block_areas is None:
                raise ValueError("structural relations require membership and block areas")
            relation_features, relation_tokens = self.relation_head(
                typed.block_tokens,
                state.block_state,
                context.block_areas,
                context.coarse_membership,
                typed.structural_mask,
                state.structural_state,
            )
            structural_tokens = structural_tokens + relation_tokens
        structural_prediction = self.structural_head(
            torch.cat(
                [
                    state.structural_state,
                    structural_tokens,
                    time_token[:, None].expand_as(structural_tokens),
                ],
                dim=-1,
            )
        )
        if self.group_moment_coupler is not None:
            if context.coarse_membership is None or state.structural_moment_state is None:
                raise ValueError("native group moments require membership and moment state")
            moment_prediction, group_residual = apply_native_group_moment(
                self.group_moment_coupler,
                typed.block_tokens,
                state.block_state,
                structural_tokens,
                state.structural_state,
                state.structural_moment_state,
                context.coarse_membership,
                typed.structural_mask,
                time,
            )
        else:
            moment_prediction = None
            group_residual = None
        port_prediction = self.port_head(
            torch.cat(
                [
                    state.port_state,
                    port_tokens,
                    time_token[:, None].expand_as(port_tokens),
                ],
                dim=-1,
            )
        ) * typed.port_mask[..., None]
        if self.use_hierarchy:
            summary = self.hierarchy_summary(
                torch.cat(
                    [state.global_state, structural_mean, port_mean],
                    dim=-1,
                )
            )
            if outline_token is not None:
                summary = summary + outline_token
            state_token = summary[:, None].expand_as(typed.block_tokens)
            latent_residual = self.hierarchy_to_latent(
                torch.cat([typed.block_tokens, state_token], dim=-1)
            )
        else:
            latent_residual = torch.zeros_like(state.block_state)
        if group_residual is not None:
            latent_residual = latent_residual + group_residual
        latent_residual = self.conditioners.residual(
            latent_residual,
            block_tokens=typed.block_tokens,
            block_state=state.block_state,
            membership=context.coarse_membership,
            structural_mask=typed.structural_mask,
            global_prediction=global_prediction,
            structural_prediction=structural_prediction,
            structural_tokens=structural_tokens,
            time=time,
            netlist=context.netlist,
            port_state=state.port_state,
            port_tokens=typed.port_tokens,
            port_mask=typed.port_mask,
        )
        (
            latent_residual,
            port_output_residual,
            port_prediction,
            atom_diagnostics,
        ) = (
            apply_hierarchy_input_adapters(
                latent_residual=latent_residual,
                block_state=state.block_state,
                block_tokens=typed.block_tokens,
                structural_tokens=structural_tokens,
                structural_mask=typed.structural_mask,
                port_state=state.port_state,
                port_prediction=port_prediction,
                port_tokens=typed.port_tokens,
                port_block_incidence=typed.port_block_incidence,
                port_mask=typed.port_mask,
                global_prediction=global_prediction,
                structural_prediction=structural_prediction,
                time=time,
                time_token=time_token,
                context=context,
                order_staging=self.order_staging,
                coarse_to_fine=self.coarse_to_fine,
                port_consensus=self.port_consensus,
                port_consensus_output=self.port_consensus_output,
                spatial_occupancy=self.spatial_occupancy,
                complexity_slot=self.complexity_slot,
                structured_heads=self.structured_heads,
                dynamic_graph_input=self.dynamic_graph_input,
                dynamic_relay=self.dynamic_relay,
                dynamic_to_latent=self.dynamic_to_latent,
            )
        )
        bridge = None
        if self.hierarchy_token_bridge:
            bridge = HierarchyTokenBridgeInput(
                block_tokens=typed.block_tokens,
                block_state=state.block_state,
                global_prediction=global_prediction,
                structural_prediction=structural_prediction,
                structural_tokens=structural_tokens,
                structural_mask=typed.structural_mask,
                time=time,
                membership=context.coarse_membership,
            )
        block_input = state.block_state + latent_residual
        if self.history_to_latent is not None:
            history = state.previous_block_state
            if history is not None:
                if history.shape != block_input.shape:
                    raise ValueError(
                        "previous_block_state must match block_state"
                    )
                block_input = block_input + self.history_to_latent(
                    history.float()
                )
        block_input, coordinate_plan = hierarchy_coordinates.parameterize_hierarchy_block_input(
            block_input, state.global_state, state.structural_state,
            context.coarse_membership, typed.structural_mask,
            enabled=self.hierarchy_coordinate_parameterization,
            cluster_local=self.hierarchy_coordinate_mode == "envelope_cluster",
            blend=self.coordinate_gate(typed.block_tokens) if self.coordinate_gate is not None else None)
        bridge_arg = {"bridge": bridge} if bridge is not None else {}
        denoiser_cond = context.denoiser_cond
        if self.state_conditioned_edge_bias is not None:
            if context.edge_state is None:
                raise ValueError("state-conditioned edges require edge cache")
            denoiser_cond = condition_state_edges(
                denoiser_cond,
                self.state_conditioned_edge_bias,
                state.block_state,
                context.edge_state.cache,
            )
        block_prediction = self.backbone(
            block_input,
            time.float(),
            denoiser_cond,
            **bridge_arg,
        )
        block_prediction = hierarchy_coordinates.restore_hierarchy_coordinate_prediction(
            block_prediction, coordinate_plan, self.coordinate_gate(typed.block_tokens) if self.coordinate_gate is not None else None)
        (
            block_prediction,
            block_base_state,
            projector_residual,
            legalizer_displacement,
            legalizer_rung_logits,
        ) = (
            apply_hierarchy_output_adapters(
                block_prediction=block_prediction,
                block_tokens=typed.block_tokens,
                block_state=state.block_state,
                time=time,
                time_token=time_token,
                context=context,
                structural_mask=typed.structural_mask,
                global_bbox_consensus=self.global_bbox_consensus,
                global_prediction=global_prediction,
                port_output_residual=port_output_residual,
                boundary_anchor=self.boundary_anchor,
                dynamic_relay=self.dynamic_relay if self.dynamic_graph_relay else None,
                projector_residual_head=self.projector_residual_head,
                structured_heads=self.structured_heads,
                apply_legalizer_response=self.structured_legalizer_apply and apply_legalizer_action,
                legalizer_response_strength=self.structured_legalizer_strength,
                legalizer_response_max_displacement=self.structured_legalizer_max_displacement,
            )
        )
        (
            block_prediction,
            legalizer_displacement,
            legalizer_rung_logits,
        ) = self.conditioners.legalizer_response(
            typed.block_tokens, state.block_state, block_prediction, time_token,
            legalizer_displacement, legalizer_rung_logits,
            self.structured_legalizer_apply and apply_legalizer_action,
            self.structured_legalizer_strength,
            self.structured_legalizer_max_displacement,
            center_mask=context.constraints[..., 1].eq(0) if context.constraints is not None else None,
        )
        return HierarchyPrediction(
            global_state=global_prediction, structural_state=structural_prediction,
            block_state=block_prediction, port_state=port_prediction,
            block_base_state=block_base_state, projector_residual=projector_residual,
            legalizer_displacement=legalizer_displacement,
            legalizer_rung_logits=legalizer_rung_logits, structural_relations=relation_features,
            structural_moment_state=moment_prediction,
            atom_diagnostics=atom_diagnostics,
        )
__all__ = [
    "HierarchicalTokenFlowModel", "HierarchyContext", "HierarchyPrediction", "HierarchyState",
    "StateEdgeContext", "build_hierarchy_targets", "outline_condition_from_boxes",
    "outline_condition_from_hw", "structural_membership"]

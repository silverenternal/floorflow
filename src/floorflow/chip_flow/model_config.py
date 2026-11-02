"""Architecture configuration serialization for chip training."""

from __future__ import annotations

import argparse


def model_config(args: argparse.Namespace) -> dict:
    if args.architecture == "trinity_token":
        return {
            "architecture": "trinity_token",
            "feature_dim": 19,
            "hidden": args.hidden,
            "depth": args.trinity_depth,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "graph_pe_k": args.graph_pe_k,
        }
    if args.architecture == "typed_hierarchy":
        return {
            "architecture": "typed_hierarchy",
            "hidden": args.hidden,
            "heads": args.heads,
            "depth": args.hierarchy_depth,
            "head_dim": args.head_dim,
            "graph_pe_k": args.graph_pe_k,
            "typed_layers": args.typed_layers,
            "typed_mode": args.typed_mode,
            "relation_specific": args.relation_specific_attention,
            "incidence_normalized_relations": (
                args.incidence_normalized_relations
            ),
            "structural_relations": args.structural_relation_head,
            "group_moment_conditioning": args.group_moment_conditioning,
            "shape_modes": getattr(args, "shape_modes", 0),
            "shape_mode_width": getattr(args, "shape_mode_width", None),
            "shape_mode_max_delta": getattr(
                args,
                "shape_mode_max_delta",
                0.5,
            ),
            "shape_mode_center_scale": getattr(
                args,
                "shape_mode_center_scale",
                0.0,
            ),
            "shape_mode_top_fractions": getattr(
                args,
                "shape_mode_top_fractions",
                None,
            ),
            "shape_mode_apply": bool(
                getattr(args, "shape_mode_apply", False)
            ),
            "shape_mode_selection": str(
                getattr(args, "shape_mode_selection", "learned")
            ),
            "complexity_conditioning": args.complexity_conditioning,
            "coarse_commit_fraction": args.coarse_commit_fraction,
            "coarse_anchor_tolerance": args.coarse_anchor_tolerance,
            "coarse_anchor_strength": args.coarse_anchor_strength,
            "coarse_anchor_max_step": args.coarse_anchor_max_step,
            "use_hierarchy": args.use_hierarchy,
            "use_netlist": args.use_netlist,
            "netlist_edge_gate": args.netlist_edge_gate,
            "dynamic_graph_relay": args.dynamic_graph_relay,
            "dynamic_graph_input": args.dynamic_graph_input,
            "projector": args.projector,
            "projector_iterations": args.projector_iterations,
            "projector_boundary_axes": args.projector_boundary_axes,
            "projector_boundary_lock_corners": args.projector_boundary_lock_corners,
            "projector_sparse_pairs": args.projector_sparse_pairs,
            "projector_strategy": args.projector_strategy,
            "projector_grid_size": args.projector_grid_size,
            "hierarchy_global_weight": args.hierarchy_global_weight,
            "hierarchy_structural_weight": args.hierarchy_structural_weight,
            "hierarchy_port_weight": args.hierarchy_port_weight,
            "hierarchy_bbox_weight": args.hierarchy_bbox_weight,
            "hierarchy_bbox_decay_steps": args.hierarchy_bbox_decay_steps,
            "outline_conditioning": args.hierarchy_outline_conditioning,
            "order_conditioning": args.hierarchy_order_conditioning,
            "coarse_to_fine_conditioning": args.hierarchy_coarse_to_fine,
            "port_consensus_conditioning": args.hierarchy_port_consensus,
            "port_consensus_output": args.hierarchy_port_output,
            "boundary_anchor_conditioning": args.hierarchy_boundary_anchor,
            "spatial_occupancy_conditioning": args.hierarchy_spatial_occupancy,
            "spatial_route_field": args.spatial_route_field,
            "symmetric_b2b_messages": args.symmetric_b2b_messages,
            "complexity_slot_conditioning": args.hierarchy_complexity_slots,
            "degree_expert_conditioning": args.degree_conditioned_expert,
            "global_spec_conditioning": getattr(
                args,
                "global_spec_conditioning",
                False,
            ),
            "node_semantic_dim": getattr(args, "node_semantic_dim", 0),
            "port_semantic_dim": getattr(args, "port_semantic_dim", 0),
            "netlist_hyperedge_conditioning": (
                getattr(args, "netlist_hyperedge_conditioning", False)
            ),
            "membership_hierarchy_conditioning": (
                getattr(args, "membership_hierarchy_conditioning", False)
            ),
            "standalone_legalizer_response": (
                getattr(args, "standalone_legalizer_response", False)
            ),
            "hierarchy_coordinate_parameterization": getattr(
                args,
                "hierarchy_coordinate_parameterization",
                False,
            ),
            "hierarchy_coordinate_mode": getattr(
                args,
                "hierarchy_coordinate_mode",
                "envelope_cluster",
            ),
            "observed_hierarchy_coupling": bool(
                getattr(args, "observed_hierarchy_coupling", False)
            ),
            "hierarchy_token_bridge": args.hierarchy_token_bridge,
            "hierarchy_token_bridge_mode": args.hierarchy_token_bridge_mode,
            "global_bbox_consensus": args.global_bbox_consensus,
            "self_conditioning": args.self_conditioning,
            "backbone_lora_rank": args.backbone_lora_rank,
            "backbone_lora_alpha": args.backbone_lora_alpha,
            "projector_residual_head": args.projector_residual_head,
            "state_conditioned_edge_bias": args.state_conditioned_edge_bias,
            "structured_heads": args.structured_heads,
            "structured_hierarchy_coupling": args.structured_hierarchy_coupling,
            "structured_factorized_block": args.structured_factorized_block,
            "structured_factorized_shared_trunk": (
                args.structured_factorized_shared_trunk
            ),
            "structured_rho_trust_radius": args.structured_rho_trust_radius,
            "structured_dual_ports": args.structured_dual_ports,
            "structured_legalizer_response": args.structured_legalizer_response,
            "structured_legalizer_apply": args.structured_legalizer_apply,
            "structured_legalizer_strength": args.structured_legalizer_strength,
            "structured_legalizer_max_displacement": (
                args.structured_legalizer_max_displacement
            ),
            "output_mode": "x0",
        }
    config = {
        "architecture": args.architecture,
        "feature_dim": (
            19 if args.architecture == "graph_slot" and args.graph_features else 10
        ),
        "hidden": args.hidden,
        "global_dim": args.global_dim,
        "target_layers": args.target_layers,
        "condition_layers": args.condition_layers,
        "decoder_layers": args.decoder_layers,
        "heads": args.heads,
        "output_mode": args.output_mode,
        "wire_dropout": args.wire_dropout,
        "constraint_dropout": args.constraint_dropout,
        "global_pool_stats": args.global_pool_stats,
        "global_film": args.global_film,
    }
    if args.architecture == "hierarchical_slot":
        config["num_slots"] = args.num_slots
    if args.architecture == "graph_slot":
        config.update(
            {
                "graph_conditioning": args.graph_bias or args.graph_messages,
                "graph_features": args.graph_features,
                "graph_bias": args.graph_bias,
                "graph_messages": args.graph_messages,
            }
        )
    if args.architecture in ("sparse_hierarchy", "slot_transport", "slot_residual"):
        config.update(
            {
                "max_slots": args.num_slots,
                "min_slots": args.min_slots,
                "assignment_top_k": args.assignment_top_k,
                "decoder_top_k": args.decoder_top_k,
                "residual_slots": args.architecture == "slot_residual",
            }
        )
    return config


__all__ = ["model_config"]

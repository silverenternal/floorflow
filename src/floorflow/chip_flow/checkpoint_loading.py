"""Load supported chip-flow checkpoints without duplicating evaluation logic."""

from __future__ import annotations

import torch

from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.hierarchical_model import ChipHierarchicalSlotFlowModel
from floorflow.chip_flow.model import ChipBlockFlowModel
from floorflow.chip_flow.shape_mode_head import attach_shape_mode_head
from floorflow.chip_flow.slot_model import ChipSlotFlowModel
from floorflow.chip_flow.sparse_flow_model import ChipSparseHierarchyFlowModel
from floorflow.chip_flow.sparse_slot_flow import ChipSlotTransportFlowModel
from floorflow.chip_flow.trinity_token_flow import ChipTrinityTokenFlowModel


def load_chip_checkpoint(
    checkpoint: dict,
    device: str | torch.device,
    *,
    use_ema: bool,
) -> torch.nn.Module:
    """Instantiate the checkpoint architecture and restore model weights."""

    config = checkpoint["config"]
    architecture = config.get(
        "architecture",
        "flat_token" if "layers" in config else "global_slot",
    )
    if architecture == "trinity_token":
        model: torch.nn.Module = ChipTrinityTokenFlowModel(
            feature_dim=int(config.get("feature_dim", 19)),
            hidden=int(config["hidden"]),
            depth=int(config.get("depth", 12)),
            heads=int(config["heads"]),
            head_dim=int(config.get("head_dim", 64)),
            graph_pe_k=int(config.get("graph_pe_k", 2)),
        )
    elif architecture == "typed_hierarchy":
        model = HierarchicalTokenFlowModel(
            hidden=int(config["hidden"]),
            heads=int(config["heads"]),
            depth=int(config.get("depth", 10)),
            head_dim=int(config.get("head_dim", 64)),
            graph_pe_k=int(config.get("graph_pe_k", 2)),
            typed_layers=int(config.get("typed_layers", 1)),
            typed_mode=config.get("typed_mode", "typed"),
            relation_specific=bool(config.get("relation_specific", False)),
            incidence_normalized_relations=bool(
                config.get("incidence_normalized_relations", False)
            ),
            structural_relations=bool(config.get("structural_relations", False)),
            group_moment_conditioning=bool(
                config.get("group_moment_conditioning", False)
            ),
            complexity_conditioning=bool(
                config.get("complexity_conditioning", False)
            ),
            coarse_commit_fraction=config.get("coarse_commit_fraction"),
            coarse_anchor_tolerance=float(
                config.get("coarse_anchor_tolerance", 0.05)
            ),
            coarse_anchor_strength=float(
                config.get("coarse_anchor_strength", 0.0)
            ),
            coarse_anchor_max_step=float(
                config.get("coarse_anchor_max_step", 0.1)
            ),
            use_hierarchy=bool(config.get("use_hierarchy", True)),
            use_netlist=bool(config.get("use_netlist", True)),
            netlist_edge_gate=bool(config.get("netlist_edge_gate", False)),
            projector=bool(config.get("projector", False)),
            projector_iterations=int(
                config.get("projector_iterations", 4)
            ),
            projector_boundary_axes=bool(
                config.get("projector_boundary_axes", False)
            ),
            projector_boundary_lock_corners=bool(
                config.get("projector_boundary_lock_corners", True)
            ),
            projector_sparse_pairs=bool(
                config.get("projector_sparse_pairs", True)
            ),
            projector_strategy=str(
                config.get("projector_strategy", "legacy")
            ),
            projector_grid_size=int(config.get("projector_grid_size", 16)),
            hierarchy_bbox_weight=float(
                config.get("hierarchy_bbox_weight", 0.0)
            ),
            hierarchy_bbox_decay_steps=int(
                config.get("hierarchy_bbox_decay_steps", 0)
            ),
            outline_conditioning=bool(
                config.get("outline_conditioning", False)
            ),
            dynamic_graph_relay=bool(
                config.get("dynamic_graph_relay", False)
            ),
            dynamic_graph_input=bool(
                config.get("dynamic_graph_input", False)
            ),
            order_conditioning=bool(
                config.get("order_conditioning", False)
            ),
            coarse_to_fine_conditioning=bool(
                config.get("coarse_to_fine_conditioning", False)
            ),
            port_consensus_conditioning=bool(
                config.get("port_consensus_conditioning", False)
            ),
            port_consensus_output=bool(
                config.get("port_consensus_output", False)
            ),
            boundary_anchor_conditioning=bool(
                config.get("boundary_anchor_conditioning", False)
            ),
            spatial_occupancy_conditioning=bool(
                config.get("spatial_occupancy_conditioning", False)
            ),
            spatial_route_field=bool(
                config.get("spatial_route_field", False)
            ),
            symmetric_b2b_messages=bool(
                config.get("symmetric_b2b_messages", False)
            ),
            complexity_slot_conditioning=bool(
                config.get("complexity_slot_conditioning", False)
            ),
            degree_expert_conditioning=bool(
                config.get("degree_expert_conditioning", False)
            ),
            global_spec_conditioning=bool(
                config.get("global_spec_conditioning", False)
            ),
            node_semantic_dim=int(config.get("node_semantic_dim", 0)),
            port_semantic_dim=int(config.get("port_semantic_dim", 0)),
            netlist_hyperedge_conditioning=bool(
                config.get("netlist_hyperedge_conditioning", False)
            ),
            membership_hierarchy_conditioning=bool(
                config.get("membership_hierarchy_conditioning", False)
            ),
            standalone_legalizer_response=bool(
                config.get("standalone_legalizer_response", False)
            ),
            observed_hierarchy_coupling=bool(
                config.get("observed_hierarchy_coupling", False)
            ),
            hierarchy_coordinate_parameterization=bool(
                config.get("hierarchy_coordinate_parameterization", False)
            ),
            hierarchy_coordinate_mode=config.get(
                "hierarchy_coordinate_mode",
                "envelope_cluster",
            ),
            hierarchy_token_bridge=bool(
                config.get("hierarchy_token_bridge", False)
            ),
            hierarchy_token_bridge_mode=config.get(
                "hierarchy_token_bridge_mode",
                "legacy",
            ),
            global_bbox_consensus=bool(
                config.get("global_bbox_consensus", False)
            ),
            self_conditioning=bool(
                config.get("self_conditioning", False)
            ),
            backbone_lora_rank=int(
                config.get("backbone_lora_rank", 0)
            ),
            backbone_lora_alpha=config.get("backbone_lora_alpha"),
            projector_residual_head=bool(
                config.get("projector_residual_head", False)
            ),
            state_conditioned_edge_bias=bool(
                config.get("state_conditioned_edge_bias", False)
            ),
            structured_heads=bool(config.get("structured_heads", False)),
            structured_hierarchy_coupling=bool(
                config.get("structured_hierarchy_coupling", True)
            ),
            structured_factorized_block=bool(
                config.get("structured_factorized_block", True)
            ),
            structured_factorized_shared_trunk=bool(
                config.get("structured_factorized_shared_trunk", False)
            ),
            structured_rho_trust_radius=float(
                config.get("structured_rho_trust_radius", 0.0)
            ),
            structured_dual_ports=bool(
                config.get("structured_dual_ports", True)
            ),
            structured_legalizer_response=bool(
                config.get("structured_legalizer_response", True)
            ),
            structured_legalizer_apply=bool(
                config.get("structured_legalizer_apply", False)
            ),
            structured_legalizer_strength=float(
                config.get("structured_legalizer_strength", 1.0)
            ),
            structured_legalizer_max_displacement=float(
                config.get("structured_legalizer_max_displacement", 0.1)
            ),
        )
        shape_modes = int(config.get("shape_modes", 0))
        if shape_modes > 0:
            attach_shape_mode_head(
                model,
                modes=shape_modes,
                width=config.get("shape_mode_width"),
                max_delta=float(config.get("shape_mode_max_delta", 0.5)),
                center_scale=float(
                    config.get("shape_mode_center_scale", 0.0)
                ),
                mode_top_fractions=config.get(
                    "shape_mode_top_fractions"
                ),
                apply=bool(config.get("shape_mode_apply", False)),
                selection=str(
                    config.get("shape_mode_selection", "learned")
                ),
            )
    elif architecture in (
        "global_slot",
        "graph_slot",
        "hierarchical_slot",
        "sparse_hierarchy",
        "slot_transport",
        "slot_residual",
    ):
        model_class = {
            "hierarchical_slot": ChipHierarchicalSlotFlowModel,
            "sparse_hierarchy": ChipSparseHierarchyFlowModel,
            "slot_transport": ChipSlotTransportFlowModel,
            "slot_residual": ChipSlotTransportFlowModel,
        }.get(architecture, ChipSlotFlowModel)
        model_kwargs = {
            "feature_dim": int(config.get("feature_dim", 10)),
            "hidden": int(config["hidden"]),
            "global_dim": int(config.get("global_dim", config["hidden"])),
            "target_layers": int(config.get("target_layers", 4)),
            "condition_layers": int(config.get("condition_layers", 2)),
            "decoder_layers": int(config.get("decoder_layers", 4)),
            "heads": int(config["heads"]),
            "output_mode": config.get("output_mode", "velocity"),
            "wire_dropout": float(config.get("wire_dropout", 0.0)),
            "constraint_dropout": float(
                config.get("constraint_dropout", 0.0)
            ),
            "global_pool_stats": bool(
                config.get("global_pool_stats", False)
            ),
            "global_film": bool(config.get("global_film", False)),
        }
        if architecture == "hierarchical_slot":
            model_kwargs["num_slots"] = int(config.get("num_slots", 8))
        if architecture == "graph_slot":
            model_kwargs.update(
                {
                    "graph_conditioning": bool(
                        config.get("graph_conditioning", True)
                    ),
                    "graph_features": bool(
                        config.get("graph_features", True)
                    ),
                    "graph_bias": bool(config.get("graph_bias", True)),
                    "graph_messages": bool(
                        config.get("graph_messages", True)
                    ),
                }
            )
        if architecture in (
            "sparse_hierarchy",
            "slot_transport",
            "slot_residual",
        ):
            model_kwargs.update(
                {
                    "max_slots": int(config.get("max_slots", 8)),
                    "min_slots": int(config.get("min_slots", 2)),
                    "assignment_top_k": int(
                        config.get("assignment_top_k", 2)
                    ),
                    "decoder_top_k": int(config.get("decoder_top_k", 4)),
                    "assignment_temperature": float(
                        config.get("assignment_temperature", 1.0)
                    ),
                    "straight_through_assignment": bool(
                        config.get("straight_through_assignment", True)
                    ),
                    "residual_slots": architecture == "slot_residual",
                }
            )
        if architecture == "slot_transport":
            model_kwargs["transport_layers"] = int(
                config.get("transport_layers", 2)
            )
        model = model_class(**model_kwargs)
    else:
        model = ChipBlockFlowModel(
            feature_dim=int(config.get("feature_dim", 10)),
            hidden=int(config["hidden"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            top_k=int(config["sparse_k"]) or None,
        )
    state = checkpoint["ema"] if use_ema else checkpoint["model"]
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = set()
    if (
        architecture == "typed_hierarchy"
        and config.get("typed_mode", "typed") != "typed_bias"
    ):
        allowed_missing.add("typed.cross_attention.type_bias")
    if config.get("port_consensus_conditioning", False) or config.get(
        "port_consensus_output",
        False,
    ) and any(key.startswith("port_consensus.") for key in state):
        allowed_missing.add("port_consensus.geometry_scale")
    if config.get("spatial_route_field", False) and any(
        key.startswith("spatial_occupancy.") for key in state
    ):
        allowed_missing.update(
            key
            for key in incompatible.missing_keys
            if key.startswith("spatial_occupancy.obstruction_encoder.")
        )
    if config.get("symmetric_b2b_messages", False) and any(
        key.startswith("typed.graph_layers.") for key in state
    ):
        allowed_missing.update(
            key
            for key in incompatible.missing_keys
            if key.startswith("typed.graph_layers.")
            and key.endswith(".symmetric_gate")
        )
    if config.get("structured_heads", False) and any(
        key.startswith("structured_heads.") for key in state
    ):
        allowed_missing.update(
            key
            for key in incompatible.missing_keys
            if key.startswith("structured_heads.atom_aggregator.")
        )
    if int(config.get("shape_modes", 0)) > 0 and any(
        key.startswith("shape_mode_head.") for key in state
    ):
        allowed_missing.update(
            key
            for key in incompatible.missing_keys
            if key.startswith(
                (
                    "shape_mode_head.mode_score.",
                    "shape_mode_head.mode_feature_encoder.",
                    "shape_mode_head.baseline_feature_score.",
                    "shape_mode_head.center_output.",
                )
            )
            or key in {
                "shape_mode_head.mode_bias",
                "shape_mode_head.baseline_bias",
            }
        )
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    new_module_prefixes = []
    if config.get("relation_specific", False):
        new_module_prefixes.append("typed.cross_attention.relation_gate")
    if config.get("structural_relations", False):
        new_module_prefixes.append("relation_head.")
    if config.get("complexity_conditioning", False):
        new_module_prefixes.append("typed.complexity.")
    if config.get("degree_expert_conditioning", False):
        new_module_prefixes.append("typed.degree_expert.")
    if config.get("global_spec_conditioning", False):
        new_module_prefixes.append("typed.global_spec_conditioner.")
    if int(config.get("node_semantic_dim", 0)) > 0:
        new_module_prefixes.append("typed.node_semantic_encoder.")
    if int(config.get("port_semantic_dim", 0)) > 0:
        new_module_prefixes.append("typed.port_semantic_encoder.")
    if config.get("netlist_hyperedge_conditioning", False):
        new_module_prefixes.append("conditioners.hyperedge.")
    if config.get("membership_hierarchy_conditioning", False):
        new_module_prefixes.append("conditioners.membership.")
    if config.get("standalone_legalizer_response", False):
        new_module_prefixes.append("conditioners.legalizer_head.")
    if config.get("hierarchy_coordinate_parameterization", False):
        new_module_prefixes.append("coordinate_gate.")
    if config.get("observed_hierarchy_coupling", False):
        new_module_prefixes.append("observed_hierarchy_coupler.")
    if config.get("dynamic_graph_input", False) or config.get(
        "dynamic_graph_relay",
        False,
    ):
        new_module_prefixes.extend(("dynamic_relay.", "dynamic_to_latent."))
    if config.get("spatial_route_field", False):
        new_module_prefixes.append("spatial_occupancy.")
    if config.get("structured_heads", False):
        new_module_prefixes.append("structured_heads.")
    if int(config.get("shape_modes", 0)) > 0:
        new_module_prefixes.append("shape_mode_head.")
    if architecture == "typed_hierarchy":
        new_module_prefixes.append("typed.local_geometry_encoder.")
    for prefix in new_module_prefixes:
        if not any(key.startswith(prefix) for key in state):
            unexpected_missing = {
                key for key in unexpected_missing if not key.startswith(prefix)
            }
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint architecture mismatch: "
            f"missing={sorted(unexpected_missing)} "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    if architecture == "typed_hierarchy":
        model.hierarchy_source_levels = config.get(
            "hierarchy_source_levels",
            "random",
        )
    return model.to(device).eval()


__all__ = ["load_chip_checkpoint"]

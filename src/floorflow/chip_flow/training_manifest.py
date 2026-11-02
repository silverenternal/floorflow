"""Training history compatibility and frozen subset loading."""

from __future__ import annotations

import json
from pathlib import Path

HISTORY_VOLATILE_KEYS = {
    "allow_full_data",
    "arm",
    "checkpoint_every",
    "eval_steps",
    "out_dir",
    "resume",
    "resume_allow_new_params",
    "resume_weights_only",
    "steps",
}

RESUME_CRITICAL_KEYS = (
    "architecture",
    "architecture_protocol",
    "training_scope",
    "training_scope_schedule",
    "require_atom_isolation",
    "stage_lr_restart",
    "stage_lr_warmup_steps",
    "typed_channel_cache_root",
    "typed_channel_cache_split",
    "sampling_mode",
    "eval_state",
    "hierarchy_source_mode",
    "hierarchy_source_noise",
    "hierarchy_source_levels",
    "hierarchy_source_shelf_probability",
    "hierarchy_source_shelf_probability_final",
    "hierarchy_source_shelf_probability_steps",
    "hierarchy_source_max_shelf_probability",
    "ema_mode",
    "ema_decay",
    "ema_warmup_steps",
    "grad_clip",
    "grad_clip_per_group",
    "checkpoint_policy",
    "backbone_unfreeze_step",
    "relation_specific_attention",
    "incidence_normalized_relations",
    "structural_relation_head",
    "structural_relation_weight",
    "group_moment_conditioning",
    "shape_modes",
    "shape_mode_width",
    "shape_mode_max_delta",
    "shape_mode_center_scale",
    "shape_mode_top_fractions",
    "shape_mode_apply",
    "shape_mode_selection",
    "shape_mode_loss_weight",
    "shape_mode_generator_weight",
    "shape_mode_diversity_weight",
    "shape_mode_usage_weight",
    "shape_mode_selection_weight",
    "observed_hierarchy_coupling",
    "shape_mode_diversity_margin",
    "shape_mode_usage_temperature",
    "complexity_conditioning",
    "dynamic_graph_input",
    "spatial_route_field",
    "coarse_commit_fraction",
    "coarse_anchor_tolerance",
    "coarse_anchor_strength",
    "coarse_anchor_max_step",
    "structured_heads",
    "structured_factorized_shared_trunk",
    "structured_rho_trust_radius",
    "structured_legalizer_apply",
    "structured_legalizer_strength",
    "structured_legalizer_max_displacement",
    "global_spec_conditioning",
    "node_semantic_dim",
    "port_semantic_dim",
    "netlist_hyperedge_conditioning",
    "membership_hierarchy_conditioning",
    "standalone_legalizer_response",
    "endpoint_compatibility_weight",
    "endpoint_compatibility_delay_steps",
    "endpoint_compatibility_anneal_steps",
    "endpoint_compatibility_iterations",
    "endpoint_compatibility_displacement_weight",
    "endpoint_compatibility_gain_weight",
    "endpoint_refiner_energy_weight",
    "endpoint_refiner_trajectory_steps",
    "endpoint_refiner_trajectory_lr",
    "endpoint_refiner_trajectory_weight",
    "hierarchy_coordinate_parameterization",
    "hierarchy_coordinate_mode",
    "endpoint_mib_share_weight",
    "endpoint_pin_tail_weight",
    "endpoint_rho_trust_weight",
    "endpoint_rho_trust_radius",
    "endpoint_pin_tail_fraction",
    "nfe_consistency_weight",
    "nfe_consistency_steps",
    "nfe_consistency_target_weight",
    "nfe_consistency_self_weight",
    "official_cost_weight",
    "official_cost_final_weight",
    "official_cost_delay_steps",
    "official_cost_anneal_steps",
    "grouping_proxy_weight",
    "grouping_proxy_final_weight",
    "grouping_proxy_delay_steps",
    "grouping_proxy_anneal_steps",
    "routing_proxy_weight",
    "routing_proxy_final_weight",
    "routing_proxy_delay_steps",
    "routing_proxy_anneal_steps",
    "legalizer_response_weight",
    "legalizer_response_delay_steps",
    "legalizer_response_anneal_steps",
    "legalizer_displacement_weight",
    "legalizer_rung_weight",
    "loss_balance_mode",
    "loss_balance_floor",
    "official_cost_violation_temperature",
    "require_deployment_source",
)

RESUME_NEW_PARAM_PREFIXES = (
    "boundary_anchor.",
    "order_staging.",
    "coarse_to_fine.",
    "port_consensus.",
    "spatial_occupancy.",
    "complexity_slot.",
    "typed.degree_expert.",
    "typed.complexity.",
    "relation_head.",
    "group_moment_coupler.",
    "observed_hierarchy_coupler.",
    "shape_mode_head.",
    "dynamic_relay.",
    "dynamic_to_latent.",
    "backbone.hierarchy_bridge.",
    "global_bbox_consensus.",
    "history_to_latent.",
    "projector_residual_head.",
    "state_conditioned_edge_bias.",
    "typed.global_spec_conditioner.",
    "typed.node_semantic_encoder.",
    "typed.port_semantic_encoder.",
    "typed.local_geometry_encoder.",
    "conditioners.hyperedge.",
    "conditioners.membership.",
    "conditioners.legalizer_head.",
    "structured_heads.",
    "coordinate_gate.",
)


def history_config(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key not in HISTORY_VOLATILE_KEYS
    }


def load_train_subset(path: Path) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("indices", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("training subset must contain a non-empty indices list")
    indices = {int(value) for value in values}
    if len(indices) != len(values):
        raise ValueError("training subset indices must be unique")
    return indices


def validate_resume_config(
    saved_config: dict,
    current_config: dict,
    *,
    weights_only: bool,
) -> None:
    """Compare resume state only when optimizer and scheduler state is restored."""

    if weights_only:
        return
    for key in RESUME_CRITICAL_KEYS:
        saved = saved_config.get(key)
        current = current_config.get(key)
        if saved is not None and current is not None and saved != current:
            raise ValueError(
                f"resume config mismatch for {key}: {saved} != {current}"
            )


__all__ = [
    "RESUME_CRITICAL_KEYS",
    "RESUME_NEW_PARAM_PREFIXES",
    "history_config",
    "load_train_subset",
    "validate_resume_config",
]

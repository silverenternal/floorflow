"""Room-instance stage execution and typed output binding."""

from __future__ import annotations

import torch

from floorflow.model.cad_layers import CadLayeredOutput


def routed_program_inputs(
    controls: dict[str, torch.Tensor],
    action,
    *,
    fuse_room_action: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Preserve the request and expose its gated CAD program token."""

    program = controls["program"]
    if action is None or not fuse_room_action:
        return program, None
    gate = action.gates["program"]
    return program, action.tokens["program"] * gate[:, None]


def apply_action_residual_gates(
    residual: torch.Tensor,
    action,
    taxonomy,
) -> torch.Tensor:
    """Apply ordered CAD gates while preserving the ungated tensor."""

    if action is None:
        return residual
    gates = action.gates
    gated = residual.clone()
    room_ids = torch.tensor(taxonomy.room_ids, device=residual.device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=residual.device)
    furniture_ids = torch.tensor(
        taxonomy.furniture_ids,
        device=residual.device,
    )
    gated[:, room_ids] *= gates["rooms"][:, None, None, None]
    wall_id = taxonomy.class_id("wall")
    gated[:, wall_id] *= gates["walls"][:, None, None]
    gated[:, opening_ids] *= gates["openings"][:, None, None, None]
    gated[:, furniture_ids] *= gates["furniture"][:, None, None, None]
    return gated


def taxonomy_class_indices(
    taxonomy,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the room, opening, and furniture index tensors once."""

    return tuple(
        torch.tensor(ids, device=device)
        for ids in (
            taxonomy.room_ids,
            taxonomy.opening_ids,
            taxonomy.furniture_ids,
        )
    )


def attach_cad_layers(
    output: dict[str, torch.Tensor],
    logits: torch.Tensor,
    taxonomy,
    room_ids: torch.Tensor,
    opening_ids: torch.Tensor,
    furniture_ids: torch.Tensor,
) -> None:
    """Bind independently ordered CAD layers to the final output."""

    wall_id = torch.tensor(
        [taxonomy.class_id("wall")],
        device=logits.device,
    )
    support_id = torch.tensor(
        [taxonomy.class_id("structure")],
        device=logits.device,
    )
    output["cad_layers"] = CadLayeredOutput(
        room_logits=logits[:, room_ids],
        wall_logits=logits[:, wall_id],
        opening_logits=logits[:, opening_ids],
        furniture_logits=logits[:, furniture_ids],
        support_logits=logits[:, support_id],
    )


def attach_room_instance(
    output: dict[str, torch.Tensor],
    stage,
    features: torch.Tensor,
    dense_room_logits: torch.Tensor,
    room_action_token: torch.Tensor,
    program: torch.Tensor,
    *,
    program_action_token: torch.Tensor | None = None,
    room_stage_candidate_logits: torch.Tensor | None = None,
    full_resolution_room_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run the room instance stage and expose all downstream contracts."""

    result = stage(
        features,
        dense_room_logits,
        room_action_token,
        program=program,
        program_action_token=program_action_token,
        room_stage_candidate_logits=room_stage_candidate_logits,
        full_resolution_room_logits=full_resolution_room_logits,
    )
    output["room_instance_logits"] = result["room_instance_logits"]
    output["room_instance_probability"] = result[
        "room_instance_probability"
    ]
    output["room_instance_class_logits"] = result["class_logits"]
    output["room_instance_mask_logits"] = result["mask_logits"]
    output["room_instance_aux_mask_logits"] = result["aux_mask_logits"]
    output["room_instance_aux_class_logits"] = result["aux_class_logits"]
    output["room_instance_query_existence_logits"] = result[
        "query_existence_logits"
    ]
    output["room_identity_tokens"] = result["query_features"]
    for key in (
        "room_owner_boundary",
        "room_owner_foreground",
        "room_owner_map",
    ):
        if key in result:
            output[key] = result[key]
    for key in (
        "room_adjacency_logits",
        "room_adjacency_mask_contact_logits",
        "room_graph_query_presence",
        "room_graph_conditioning_scale",
        "room_graph_second_pass_scale",
        "room_graph_first_class_residual",
        "room_graph_first_mask_residual",
        "room_graph_second_class_residual",
        "room_graph_second_mask_residual",
        "room_graph_first_owner_swap_fraction",
        "room_graph_first_owner_swap_support_fraction",
        "room_graph_second_owner_swap_fraction",
        "room_graph_second_owner_swap_support_fraction",
        "slot_owner_probability",
        "mask_class_evidence",
        "query_existence_logits",
        "room_count_logits",
        "room_presence_count_logits",
        "room_class_count_logits",
        "query_set_diagnostics",
        "room_class_count_prior_scale",
        "room_stage_prior_scale",
        "query_count_candidate_mask",
        "query_count_selection_mask",
        "query_count_soft_selection",
        "query_count_ordered_indices",
        "room_query_selected_count",
        "room_owner_soft_weight_mean",
        "room_owner_relative_confidence_mean",
        "room_owner_foreground_fraction",
        "room_owner_mass_sum_mean",
        "room_owner_fused_logits",
        "room_query_residual_abs_mean",
        "room_full_resolution_mask_scale",
        "program_room_assignment_logits",
        "program_room_class_logits",
        "program_room_scheduled_logits",
        "program_room_scheduled_class_logits",
        "adaptive_query_diagnostics",
    ):
        if key in result:
            output[key] = result[key]
    return result


__all__ = [
    "apply_action_residual_gates",
    "attach_cad_layers",
    "attach_room_instance",
    "routed_program_inputs",
    "taxonomy_class_indices",
]

"""Auditable forward-stage contract for raster instance controls."""

from __future__ import annotations


def raster_instance_module_contract(model) -> dict[str, object]:
    """Return the active module order without changing execution."""

    stages = []
    if model.dual_control:
        stages.append("dual_control_projection")
    stages.extend(["room_encoder", "dense_room_head"])
    if model.dense_identity_refiner is not None:
        stages.append("dense_identity_refiner")
    if model.room_program_head is not None:
        stages.append("room_program_head")
    stages.append("half_resolution_query_decoder")
    if model.mask_refiner is not None:
        stages.append("full_resolution_mask_refiner")
    if model.room_boundary_owner_refiner is not None:
        stages.append("room_boundary_owner_refiner")
    if model.mask_pooled_class_fusion is not None:
        stages.append("mask_pooled_class_fusion")
    if model.full_resolution_class_sampler is not None:
        stages.append("full_resolution_class_sampler")
    if model.joint_room_slot_state is not None:
        stages.append("joint_room_slot_state")
    if (
        model.room_program_head is not None
        and model.program_class_prior_weight > 0.0
    ):
        stages.append("program_class_prior")
    if model.query_set_head is not None:
        stages.append("query_set_head")
        if model.query_set_head.set_attention:
            stages.append("query_set_attention")
        stages.append(
            "soft_count_selection"
            if model.query_count_gate
            else "presence_only"
        )
    stages.append(
        "owner_semantic_reassignment"
        if model.owner_semantic_reassignment is not None
        else "class_mass_fusion"
    )
    if model.owner_semantic_reassignment is None:
        stages.append("full_resolution_room_readout")
    stages.append("query_owner_readout")
    return {
        "ordered_stages": stages,
        "query_count_gate": model.query_count_gate,
        "train_count_routing": (
            "soft"
            if model.query_count_gate and model.query_set_head is not None
            else "disabled"
        ),
        "eval_count_routing": (
            "hard"
            if model.query_count_gate and model.query_set_head is not None
            else "disabled"
        ),
    }


__all__ = ["raster_instance_module_contract"]

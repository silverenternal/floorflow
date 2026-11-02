"""Training controls for raster instance-query room decoders."""

from __future__ import annotations

from floorflow.heads.raster_control_bank import RasterInstanceControl


def configure_room_class_recovery(
    model: RasterInstanceControl,
) -> list[str]:
    """Freeze proposal geometry and train only query-to-class parameters."""

    model.requires_grad_(False)
    modules = (
        model.mask_decoder.class_fusion,
        model.mask_decoder.class_head,
        model.mask_decoder.decoder_norm,
        *(
            [model.mask_decoder.adaptive_query_refiner]
            if model.mask_decoder.adaptive_query_refiner is not None
            else []
        ),
        *(
            [model.mask_pooled_class_fusion]
            if model.mask_pooled_class_fusion is not None
            else []
        ),
        *(
            [model.joint_room_slot_state]
            if model.joint_room_slot_state is not None
            else []
        ),
        *(
            [model.dense_identity_refiner]
            if model.dense_identity_refiner is not None
            else []
        ),
        *(
            [model.room_program_head]
            if model.room_program_head is not None
            else []
        ),
    )
    for module in modules:
        module.requires_grad_(True)
    trained_prefixes = (
        "mask_decoder.class_fusion.",
        "mask_decoder.class_head.",
        "mask_decoder.decoder_norm.",
        "mask_decoder.adaptive_query_refiner.",
        "mask_pooled_class_fusion.",
        "joint_room_slot_state.",
        "dense_identity_refiner.",
        "room_program_head.",
    )
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith(trained_prefixes)
    ]


def configure_joint_slot_state_recovery(
    model: RasterInstanceControl,
) -> list[str]:
    """Freeze the parent model and train only the joint slot-state head."""

    if model.joint_room_slot_state is None:
        raise ValueError("joint slot-state recovery requires the slot-state head")
    model.requires_grad_(False)
    model.joint_room_slot_state.requires_grad_(True)
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


__all__ = [
    "configure_joint_slot_state_recovery",
    "configure_room_class_recovery",
]

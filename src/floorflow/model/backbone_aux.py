"""Auxiliary head composition for the floorplan flow backbone."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.heads.room_groups import room_class_group_index


def run_forward_with_aux(
    model,
    x_t: torch.Tensor,
    t: torch.Tensor,
    footprint: torch.Tensor,
    floor_ids: torch.Tensor | None = None,
    detach_object_features: bool = False,
    layout_prior_logits: torch.Tensor | None = None,
    face_ids: torch.Tensor | None = None,
    room_program: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output_features = model._encode_features(x_t, t, footprint, floor_ids=floor_ids)
    velocity = model._velocity_from_features(output_features, t)
    semantic_logits = (
        model.semantic_endpoint(output_features)
        if model.use_semantic_endpoint_head
        else None
    )
    room_prior = None
    if model.use_room_partition_semantic_prior and semantic_logits is not None:
        room_prior = semantic_logits[:, model.room_class_ids.to(semantic_logits.device) - 1]
    hierarchical_predictions = (
        model.hierarchical_rooms(
            output_features,
            room_prior,
            footprint,
            layout_prior_logits=layout_prior_logits,
        )
        if model.use_hierarchical_rooms
        else None
    )
    room_partition_logits = (
        model.room_partition(output_features, room_prior)
        if model.use_room_partition_head
        else None
    )
    program_condition = None
    if (
        room_partition_logits is not None
        and room_program is not None
        and model.use_program_conditioning
    ):
        program_condition = model.program_conditioner(room_program)
        room_partition_logits = (
            room_partition_logits
            + model.program_scale
            * program_condition["class_bias"][:, :, None, None]
        )
    room_program_predictions = None
    if hierarchical_predictions is not None:
        room_program_predictions = hierarchical_predictions["program"]
    elif model.use_room_program_head and model.num_room_classes > 0:
        room_program_predictions = model.room_program(output_features)
    room_group_logits = (
        model.room_group(output_features) if model.use_room_group_head else None
    )
    if room_partition_logits is not None and room_group_logits is not None:
        class_groups = room_class_group_index(
            model.semantic_taxonomy,
            room_partition_logits.device,
        )
        room_partition_logits = (
            room_partition_logits
            + model.room_group_scale * room_group_logits[:, class_groups]
        )
    if room_partition_logits is not None and room_program_predictions is not None:
        program_bias = (
            room_program_predictions["class_bias"]
            + room_program_predictions["presence_logits"]
        )
        program_scale = (
            model.hierarchical_program_scale
            if hierarchical_predictions is not None
            else model.room_program_scale
        )
        room_partition_logits = (
            room_partition_logits + program_scale * program_bias[:, :, None, None]
        )
    if hierarchical_predictions is not None and room_partition_logits is not None:
        coarse = F.interpolate(
            hierarchical_predictions["refined_room_logits"],
            size=room_partition_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        semantic = F.interpolate(
            hierarchical_predictions["semantic_bias"],
            size=room_partition_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        room_partition_logits = (
            room_partition_logits
            + model.coarse_room_scale.to(room_partition_logits.dtype) * coarse
            + model.hierarchical_scale.to(room_partition_logits.dtype) * semantic
        )
    if model.use_coarse_to_face_cfm and room_partition_logits is not None:
        flow_semantics = room_partition_logits
        semantic_tokens = None
        if hierarchical_predictions is not None:
            flow_semantics = F.interpolate(
                hierarchical_predictions["refined_room_logits"],
                size=room_partition_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            semantic_tokens = hierarchical_predictions["room_graph_tokens"]
        velocity = velocity + model.coarse_to_face_cfm(
            x_t,
            t,
            output_features,
            footprint,
            flow_semantics,
            face_ids,
            semantic_tokens,
        )
    polygon_loop_predictions = None
    if model.use_polygon_loop_head:
        class_tokens = None
        if hierarchical_predictions is not None:
            class_tokens = hierarchical_predictions["room_graph_tokens"]
        elif room_program_predictions is not None:
            class_tokens = room_program_predictions["class_tokens"]
        polygon_loop_predictions = model.polygon_loop(output_features, class_tokens)
    room_mask_predictions = None
    if model.use_room_mask_head:
        low_resolution = F.adaptive_avg_pool2d(
            output_features,
            (model.room_mask_resolution, model.room_mask_resolution),
        )
        if room_partition_logits is not None and model.room_mask_use_partition:
            low_resolution = torch.cat(
                [
                    low_resolution,
                    F.interpolate(
                        room_partition_logits,
                        size=(model.room_mask_resolution, model.room_mask_resolution),
                        mode="bilinear",
                        align_corners=False,
                    ),
                ],
                dim=1,
            )
        room_mask_predictions = model.room_mask_decoder(low_resolution)
        if room_partition_logits is not None and model.room_mask_use_partition:
            room_mask_predictions["prior_logits"] = room_partition_logits
        room_mask_predictions["mask_logits"] = F.interpolate(
            room_mask_predictions["mask_logits"],
            size=output_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        room_mask_predictions["aux_mask_logits"] = [
            F.interpolate(
                logits,
                size=output_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for logits in room_mask_predictions.get("aux_mask_logits", [])
        ]
    object_features = (
        output_features.detach() if detach_object_features else output_features
    )
    auxiliary = {
        "topology": model.topology_aux(output_features) if model.use_topology_aux else {},
        "semantic_logits": semantic_logits,
        "room_partition_logits": room_partition_logits,
        "room_program": room_program_predictions,
        "room_group_logits": room_group_logits,
        "hierarchical_rooms": hierarchical_predictions,
        "polygon_loops": polygon_loop_predictions,
        "room_mask_predictions": room_mask_predictions,
        "structured_objects": (
            model.structured_objects(object_features)
            if model.use_structured_object_head
            else None
        ),
        "geometry": (
            model.geometry_heads(output_features)
            if model.use_geometry_heads
            else None
        ),
        "family_stream": getattr(model, "_last_family_diagnostics", None),
        "program_condition": program_condition,
    }
    return velocity, auxiliary


__all__ = ["run_forward_with_aux"]

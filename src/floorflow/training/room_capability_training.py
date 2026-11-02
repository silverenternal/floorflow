"""Joint instance and dense losses for the room capability stage."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.room_mask_decoder import (
    room_mask_loss,
    room_mask_partition_loss,
    room_mask_targets,
)
from floorflow.heads.room_query_set_head import room_query_set_loss
from floorflow.objectives.room_adjacency import (
    dense_room_type_graph_losses,
    room_adjacency_consistency_loss,
    room_type_adjacency_consistency_loss,
    room_type_node_consistency_loss,
)
from floorflow.objectives.room_query_matching import match_room_mask_queries
from floorflow.training.program_instance_training import (
    program_instance_assignment_loss,
)
from floorflow.training.proposal_distillation import TeacherTargetBatch


def instance_room_losses(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    max_rooms: int,
    teacher: TeacherTargetBatch | None = None,
    program: torch.Tensor | None = None,
    program_assignment_weight: float = 0.2,
    room_adjacency_weight: float = 0.1,
    room_type_adjacency_weight: float = 0.0,
    room_type_node_weight: float = 0.0,
    dense_type_adjacency_weight: float = 0.0,
    dense_type_node_weight: float = 0.0,
    dense_type_edge_overlap_weight: float = 0.0,
    dense_type_node_overlap_weight: float = 0.0,
    query_set_weight: float = 0.5,
    query_class_cardinality_weight: float = 0.1,
    query_support_assignment_weight: float = 0.0,
    query_selection_alignment_weight: float = 0.0,
    query_presence_rank_weight: float = 0.0,
    query_presence_count_weight: float = 0.0,
    unmatched_mask_weight: float = 0.0,
    mask_pos_weight: float = 1.0,
    mask_tversky_weight: float = 0.0,
    query_separation_weight: float = 0.0,
    query_separation_class_aware: bool = True,
    reference_anchor_weight: float = 0.0,
    mask_centroid_weight: float = 0.0,
    balance_room_classes: bool = True,
    return_loss_terms: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]] | tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Supervise query instances and their dense fused partition."""

    if teacher is None:
        target_classes, target_masks, target_valid = room_mask_targets(
            target,
            taxonomy,
            max_rooms=int(max_rooms),
        )
    else:
        target_classes, target_masks, target_valid = teacher_room_targets(
            teacher,
            target,
            taxonomy,
            max_rooms=int(max_rooms),
        )
    mask_logits = F.interpolate(
        output["room_instance_mask_logits"],
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    predictions = {
        "class_logits": output["room_instance_class_logits"],
        "mask_logits": mask_logits,
        "query_existence_logits": output.get("query_existence_logits"),
        "query_count_selection_mask": output.get(
            "query_count_selection_mask"
        ),
        "aux_mask_logits": [
            F.interpolate(
                value,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for value in output["room_instance_aux_mask_logits"]
        ],
        "aux_class_logits": output["room_instance_aux_class_logits"],
        "adaptive_query_diagnostics": output.get(
            "adaptive_query_diagnostics"
        ),
    }
    room_matches = match_room_mask_queries(
        predictions["class_logits"],
        predictions["mask_logits"],
        target_classes,
        target_masks,
        target_valid,
    )
    room_shared_match_count = mask_logits.new_tensor(
        sum(len(match.predicted_indices) for match in room_matches)
    )
    room_valid_target_count = target_valid.sum().to(mask_logits.dtype)
    room_query_gate_output_present = mask_logits.new_tensor(
        float("query_count_soft_selection" in output)
    )
    room_class_weights = (
        _room_class_weights(taxonomy, target.device)
        if balance_room_classes
        else None
    )
    instance_loss, instance_components = room_mask_loss(
        predictions,
        target_classes,
        target_masks,
        target_valid,
        class_weights=room_class_weights,
        boundary_weight=2.0,
        unmatched_mask_weight=unmatched_mask_weight,
        mask_pos_weight=mask_pos_weight,
        mask_tversky_weight=mask_tversky_weight,
        query_separation_weight=(
            1.0 if float(query_separation_weight) > 0.0 else 0.0
        ),
        class_aware_query_separation=query_separation_class_aware,
        reference_anchor_weight=reference_anchor_weight,
        mask_centroid_weight=mask_centroid_weight,
        matches=room_matches,
    )
    query_separation_loss = instance_components["_room_mask_separation_term"]
    base_instance_loss = instance_loss - query_separation_loss
    instance_loss = (
        base_instance_loss
        + float(query_separation_weight) * query_separation_loss
    )
    query_set_total = instance_loss.new_zeros(())
    query_set_components = {}
    query_set_loss_terms: list[dict[str, torch.Tensor]] = []
    if "room_count_logits" in output:
        query_predictions = {
            **predictions,
            **{
                key: output[key]
                for key in (
                    "room_count_logits",
                    "room_class_count_logits",
                    "query_existence_logits",
                    "slot_owner_probability",
                    "query_count_soft_selection",
                    "query_count_selection_mask",
                )
                if key in output
            },
        }
        query_set_result = room_query_set_loss(
            query_predictions,
            target_classes,
            target_valid,
            target_masks,
            class_cardinality_consistency_weight=(
                query_class_cardinality_weight
            ),
            support_assignment_weight=query_support_assignment_weight,
            selection_alignment_weight=query_selection_alignment_weight,
            presence_rank_weight=query_presence_rank_weight,
            presence_count_regression_weight=query_presence_count_weight,
            matches=room_matches,
            return_loss_terms=return_loss_terms,
        )
        (
            query_set_total,
            query_set_components,
            *query_set_loss_terms,
        ) = query_set_result
        instance_loss = (
            instance_loss + float(query_set_weight) * query_set_total
        )
    partition_loss, partition_components = room_mask_partition_loss(
        predictions,
        target,
        footprint,
        taxonomy,
        class_weights=room_class_weights,
    )
    program_loss = instance_loss.new_zeros(())
    program_components = {}
    if (
        program is not None
        and "program_room_assignment_logits" in output
    ):
        program_loss, program_components = (
            program_instance_assignment_loss(
                output["program_room_assignment_logits"],
                mask_logits,
                target_classes,
                target_masks,
                target_valid,
                program,
                matches=room_matches,
            )
        )
        instance_loss = (
            instance_loss + float(program_assignment_weight) * program_loss
        )
    adjacency_loss = instance_loss.new_zeros(())
    adjacency_components = {}
    type_adjacency_loss = instance_loss.new_zeros(())
    type_adjacency_components = {}
    type_node_loss = instance_loss.new_zeros(())
    type_node_components = {}
    if "room_adjacency_logits" in output:
        adjacency_loss, adjacency_components = (
            room_adjacency_consistency_loss(
                output,
                int(max_rooms),
                target_masks,
                target_valid,
            )
        )
        instance_loss = (
            instance_loss + float(room_adjacency_weight) * adjacency_loss
        )
        type_adjacency_loss, type_adjacency_components = (
            room_type_adjacency_consistency_loss(
                output,
                int(max_rooms),
                target_classes,
                target_valid,
                target_masks,
            )
        )
        instance_loss = (
            instance_loss
            + float(room_type_adjacency_weight) * type_adjacency_loss
        )
    if float(room_type_node_weight) > 0.0:
        type_node_loss, type_node_components = (
            room_type_node_consistency_loss(
                output,
                int(max_rooms),
                target_classes,
                target_valid,
            )
        )
        instance_loss = (
            instance_loss + float(room_type_node_weight) * type_node_loss
        )
    dense_components = {}
    dense_total = instance_loss.new_zeros(())
    dense_loss_terms: list[dict[str, torch.Tensor]] = []
    if (
        float(dense_type_adjacency_weight) > 0.0
        or float(dense_type_node_weight) > 0.0
    ) and "room_instance_probability" in output:
        final_room_probability = output["room_instance_probability"]
        if "room_instance_logits" in output:
            final_room_probability = output[
                "room_instance_logits"
            ].float().softmax(dim=1)
        dense_result = dense_room_type_graph_losses(
            final_room_probability,
            target,
            taxonomy,
            adjacency_weight=dense_type_adjacency_weight,
            node_weight=dense_type_node_weight,
            adjacency_overlap_weight=dense_type_edge_overlap_weight,
            node_overlap_weight=dense_type_node_overlap_weight,
            return_loss_terms=return_loss_terms,
        )
        dense_total, dense_components, *dense_loss_terms = dense_result
        instance_loss = instance_loss + dense_total
    components = {
        **instance_components,
        "instance_room_class": instance_components[
            "room_mask_class_loss"
        ],
        "instance_room_mask": instance_components[
            "room_mask_dice_bce_loss"
        ],
        "instance_room_unmatched": instance_components[
            "room_mask_unmatched_loss"
        ],
        "instance_room_tversky": instance_components[
            "room_mask_tversky_loss"
        ],
        "instance_room_matched_dice": instance_components[
            "room_mask_matched_dice"
        ],
        "instance_room_matched_iou": instance_components[
            "room_mask_matched_iou"
        ],
        "instance_room_predicted_area": instance_components[
            "room_mask_matched_predicted_area"
        ],
        "instance_room_target_area": instance_components[
            "room_mask_matched_target_area"
        ],
        "instance_room_unmatched_area": instance_components[
            "room_mask_unmatched_predicted_area"
        ],
        "instance_room_centroid_error": instance_components[
            "room_mask_matched_centroid_error"
        ],
        "instance_room_reference_anchor": instance_components[
            "room_mask_reference_anchor_loss"
        ],
        "instance_room_mask_centroid": instance_components[
            "room_mask_centroid_loss"
        ],
        "instance_room_query_separation": instance_components[
            "room_query_separation_loss"
        ],
        "instance_room_partition": partition_loss.detach(),
        "program_instance_total": program_loss.detach(),
        "room_query_set_total": query_set_total.detach(),
        **query_set_components,
        "room_shared_match_count": room_shared_match_count,
        "room_valid_target_count": room_valid_target_count,
        "room_query_gate_output_present": room_query_gate_output_present,
        "room_adjacency_total": adjacency_loss.detach(),
        **adjacency_components,
        "room_type_adjacency_total": type_adjacency_loss.detach(),
        **type_adjacency_components,
        "room_type_node_total": type_node_loss.detach(),
        **type_node_components,
        **dense_components,
        **program_components,
        **partition_components,
    }
    if return_loss_terms:
        raw_terms = {
            "instance_room": base_instance_loss,
            "query_separation": query_separation_loss,
            "program_instance": program_loss,
            "room_adjacency": adjacency_loss,
            "room_type_adjacency": type_adjacency_loss,
            "room_type_node": type_node_loss,
        }
        if query_set_loss_terms:
            raw_terms.update(query_set_loss_terms[0])
        if dense_loss_terms:
            raw_terms.update(dense_loss_terms[0])
        return (
            instance_loss,
            partition_loss,
            components,
            raw_terms,
        )
    return instance_loss, partition_loss, components


def _room_class_weights(
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
) -> torch.Tensor:
    """Balance rare room types from provenance counts, not batch labels."""

    counts = taxonomy.source_counts.get("room", {})
    frequencies = [
        float(
            counts.get(
                taxonomy.class_names[class_id].removeprefix("room:"),
                0,
            )
        )
        for class_id in taxonomy.room_ids
    ]
    values = torch.tensor(frequencies, device=device, dtype=torch.float32)
    if float(values.sum()) <= 0.0:
        return torch.ones_like(values)
    weights = torch.rsqrt(values.clamp_min(1.0))
    weights = weights / weights.mean().clamp_min(1.0e-8)
    return weights.clamp(0.25, 4.0)


def teacher_room_targets(
    teacher: TeacherTargetBatch,
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    max_rooms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign room classes to class-agnostic Raster2Seq proposal masks."""

    count = min(int(max_rooms), teacher.masks.shape[1])
    masks = F.interpolate(
        teacher.masks[:, :count].float(),
        size=target.shape[-2:],
        mode="nearest",
    ) > 0.5
    teacher_classes = teacher.class_ids
    if teacher_classes is None:
        room_proposals = teacher.valid[:, :count].bool()
    else:
        teacher_classes = teacher_classes[:, :count]
        room_ids_tensor = torch.tensor(
            taxonomy.room_ids,
            device=target.device,
        )
        room_proposals = torch.isin(teacher_classes, room_ids_tensor)
    valid = (
        room_proposals
        & masks.flatten(2).any(dim=-1)
    )
    classes = torch.zeros(
        valid.shape,
        dtype=torch.long,
        device=target.device,
    )
    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=target.device,
    )
    for batch_index in range(target.shape[0]):
        for query in range(count):
            if not bool(valid[batch_index, query]):
                continue
            values = target[batch_index][masks[batch_index, query]]
            counts = torch.bincount(
                values,
                minlength=taxonomy.num_classes,
            )
            room_counts = counts[room_ids]
            if int(room_counts.max()) == 0:
                valid[batch_index, query] = False
                continue
            classes[batch_index, query] = int(room_counts.argmax())
    return classes, masks.float(), valid


__all__ = [
    "instance_room_losses",
    "teacher_room_targets",
]

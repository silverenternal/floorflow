"""Single-source argument binding for predicted-control instance losses."""

from __future__ import annotations


def typed_channel_instance_loss_kwargs(
    args,
    *,
    return_loss_terms: bool,
) -> dict[str, object]:
    """Map parsed training arguments to instance_room_losses keywords."""

    return {
        "program_assignment_weight": (
            args.program_instance_assignment_weight
        ),
        "room_adjacency_weight": args.room_instance_adjacency_weight,
        "room_type_adjacency_weight": args.room_type_adjacency_weight,
        "room_type_node_weight": args.room_type_node_weight,
        "dense_type_adjacency_weight": args.dense_type_adjacency_weight,
        "dense_type_node_weight": args.dense_type_node_weight,
        "dense_type_edge_overlap_weight": (
            args.dense_type_edge_overlap_weight
        ),
        "dense_type_node_overlap_weight": (
            args.dense_type_node_overlap_weight
        ),
        "query_set_weight": args.room_query_set_loss_weight,
        "query_class_cardinality_weight": (
            args.room_query_class_cardinality_weight
        ),
        "query_support_assignment_weight": (
            args.room_query_support_assignment_weight
        ),
        "query_selection_alignment_weight": (
            args.room_query_selection_alignment_weight
        ),
        "query_presence_rank_weight": (
            args.room_query_presence_rank_weight
        ),
        "query_presence_count_weight": (
            args.room_query_presence_count_weight
        ),
        "unmatched_mask_weight": args.room_instance_unmatched_mask_weight,
        "mask_pos_weight": args.room_instance_mask_pos_weight,
        "mask_tversky_weight": args.room_instance_mask_tversky_weight,
        "query_separation_weight": (
            args.room_instance_query_separation_weight
        ),
        "query_separation_class_aware": (
            not args.room_instance_query_separation_class_agnostic
        ),
        "reference_anchor_weight": (
            args.room_instance_reference_anchor_weight
        ),
        "mask_centroid_weight": args.room_instance_mask_centroid_weight,
        "balance_room_classes": args.room_instance_class_balance,
        "return_loss_terms": bool(return_loss_terms),
    }


__all__ = ["typed_channel_instance_loss_kwargs"]

"""Loss-term registry for typed predicted-control training."""

from __future__ import annotations

from floorflow.objectives.gradient_balancer import GradientNormBalancer


def _balance_reference(scope: str, requested: str) -> str:
    """Use a gate-driving loss as the anchor for each training scope."""

    if requested != "auto":
        return requested
    if scope.startswith("room"):
        return "instance_room"
    if scope.startswith("wall"):
        return "stage_wall"
    if scope.startswith("opening"):
        return "stage_opening"
    if scope.startswith("furniture"):
        return "stage_furniture"
    return "instance_room"


def build_typed_channel_balancer(args):
    """Create a balancer covering every independently optimized objective."""

    if not bool(args.dynamic_loss_weights):
        return None
    instance_scale = float(args.instance_room_weight)
    query_scale = (
        instance_scale * float(args.room_query_set_loss_weight)
    )
    reference = _balance_reference(
        str(args.train_scope),
        str(getattr(args, "dynamic_balance_reference", "auto")),
    )
    return GradientNormBalancer(
        {
            "ce": args.ce_weight,
            "room_iou": args.room_iou_weight,
            "tversky": args.tversky_weight,
            "boundary": args.boundary_weight,
            "family_iou": args.family_iou_weight,
            "routing": args.routing_regularization_weight,
            "furniture_query": args.furniture_query_weight,
            "stage_wall": args.stage_family_weight,
            "stage_wall_precision": (
                args.stage_family_weight * args.wall_precision_weight
            ),
            "stage_opening": args.stage_family_weight,
            "stage_opening_precision": (
                args.stage_family_weight * args.opening_precision_weight
            ),
            "stage_opening_host": (
                args.stage_family_weight * args.opening_host_weight
            ),
            "stage_opening_type": 0.5 * args.stage_family_weight,
            "stage_furniture": 0.5 * args.stage_family_weight,
            "instance_room": instance_scale,
            "query_separation": (
                instance_scale
                * float(args.room_instance_query_separation_weight)
            ),
            "instance_partition": args.instance_room_partition_weight,
            "program_instance": (
                float(args.program_instance_assignment_weight)
                * instance_scale
            ),
            "room_adjacency": (
                instance_scale * args.room_instance_adjacency_weight
            ),
            "room_type_adjacency": (
                instance_scale * args.room_type_adjacency_weight
            ),
            "room_type_node": (
                instance_scale * args.room_type_node_weight
            ),
            "dense_type_adjacency": (
                instance_scale * args.dense_type_adjacency_weight
            ),
            "dense_type_node": (
                instance_scale * args.dense_type_node_weight
            ),
            "query_existence": 0.5 * query_scale,
            "query_count": 0.5 * query_scale,
            "query_class_count": 0.1 * query_scale,
            "query_class_cardinality": (
                float(args.room_query_class_cardinality_weight)
                * query_scale
            ),
            "query_support_assignment": (
                float(args.room_query_support_assignment_weight)
                * query_scale
            ),
            "query_selection_alignment": (
                float(args.room_query_selection_alignment_weight)
                * query_scale
            ),
            "query_presence_rank": (
                float(args.room_query_presence_rank_weight)
                * query_scale
            ),
            "query_presence_count": (
                float(args.room_query_presence_count_weight)
                * query_scale
            ),
            "furniture_assignment": args.furniture_room_assignment_weight,
            "furniture_containment": args.furniture_containment_weight,
        },
        reference=reference,
        update_interval=int(args.dynamic_balance_update_interval),
        exponent=float(args.dynamic_balance_exponent),
    )


__all__ = ["build_typed_channel_balancer"]

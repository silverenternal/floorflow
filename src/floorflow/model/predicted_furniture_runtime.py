"""Furniture-stage execution and output binding for predicted CAD controls."""

from __future__ import annotations

import torch

from floorflow.model.predicted_action_context import maybe_detach


def attach_furniture_queries(
    output: dict[str, torch.Tensor],
    stage,
    structure_features: torch.Tensor,
    controls: dict[str, torch.Tensor],
    footprint: torch.Tensor,
    action_token: torch.Tensor,
    room_logits: torch.Tensor,
    detach_backbone: bool,
) -> None:
    """Run the final furniture stage and bind its typed outputs."""

    query_features = maybe_detach(structure_features, detach_backbone)
    furniture_queries = stage(
        query_features,
        controls,
        footprint,
        stage_token=maybe_detach(action_token, detach_backbone),
        room_logits=maybe_detach(room_logits, detach_backbone),
    )
    output["furniture_query_class_logits"] = furniture_queries["class_logits"]
    output["furniture_query_mask_logits"] = furniture_queries["mask_logits"]
    output["furniture_query_budget"] = furniture_queries["query_budget"]
    output["furniture_query_active"] = furniture_queries["active_queries"]
    output["furniture_query_room_probability"] = furniture_queries[
        "furniture_query_room_probability"
    ]
    if "room_assignment_logits" in furniture_queries:
        output["furniture_room_assignment_logits"] = furniture_queries[
            "room_assignment_logits"
        ]
        output["furniture_query_raw_mask_logits"] = furniture_queries[
            "raw_mask_logits"
        ]


__all__ = ["attach_furniture_queries"]

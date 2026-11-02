"""Build base and graph-conditioned room priors for overlay evaluation."""

from __future__ import annotations

import torch

from floorflow.inference.recursive_room_prior import (
    predict_recursive_room_prior,
)


def predict_base_and_graph_priors(
    *,
    segmenter,
    policy,
    graph_adapter,
    footprint,
    direct_prior,
    args,
    room_group_index,
):
    recursive = predict_recursive_room_prior(
        policy,
        segmenter,
        footprint,
        max_depth=args.recursive_max_depth,
        min_leaf_pixels=args.recursive_min_leaf_pixels,
        context_fraction=args.recursive_context_fraction,
        global_logits=direct_prior,
        recursive_blend=args.recursive_blend,
        graph_blend=args.recursive_graph_blend,
        graph_steps=args.recursive_graph_steps,
        graph_margin_threshold=args.recursive_graph_margin,
        room_group_index=room_group_index,
        graph_adapter=(
            None
            if graph_adapter is not None and args.room_gate_decouple
            else graph_adapter
        ),
    )
    base_prior = torch.where(
        (recursive.action == 0)[:, None, None, None],
        direct_prior,
        recursive.logits,
    )
    graph_prior = None
    decision = {
        "actions": recursive.action.cpu().tolist(),
        "strengths": recursive.recursion_strength.cpu().tolist(),
        "depths": [
            [region.depth for region in regions]
            for regions in recursive.regions
        ],
        "graph": list(recursive.graph_diagnostics),
    }
    if graph_adapter is not None and args.room_gate_decouple:
        graph_recursive = predict_recursive_room_prior(
            policy,
            segmenter,
            footprint,
            max_depth=args.recursive_max_depth,
            min_leaf_pixels=args.recursive_min_leaf_pixels,
            context_fraction=args.recursive_context_fraction,
            global_logits=direct_prior,
            recursive_blend=args.recursive_blend,
            graph_blend=args.recursive_graph_blend,
            graph_steps=args.recursive_graph_steps,
            graph_margin_threshold=args.recursive_graph_margin,
            room_group_index=room_group_index,
            graph_adapter=graph_adapter,
        )
        graph_prior = torch.where(
            (graph_recursive.action == 0)[:, None, None, None],
            direct_prior,
            graph_recursive.logits,
        )
        decision["graph_gated"] = list(graph_recursive.graph_diagnostics)
    return base_prior, graph_prior, decision


__all__ = ["predict_base_and_graph_priors"]

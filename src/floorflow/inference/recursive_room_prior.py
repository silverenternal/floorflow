"""Recursive target-free room priors gated by footprint complexity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from floorflow.heads.room_segmenter import ContourRoomSegmenter
from floorflow.recursion.budget import allocate_dynamic_budgets
from floorflow.recursion.complexity import (
    POLICY_ACTION_NAMES,
    POLICY_INPUT_FEATURE_NAMES,
    RecursionPolicy,
    complexity_features,
)
from floorflow.recursion.graph import (
    build_partition_graph,
    graph_refine_region_logits,
)
from floorflow.recursion.graph_adapter import RecursiveSemanticGraphAdapter
from floorflow.recursion.partition import (
    PartitionRegion,
    merge_region_logits,
    propose_partition,
)


@dataclass(frozen=True)
class RecursiveRoomPriorOutput:
    logits: torch.Tensor
    action: torch.Tensor
    recursion_strength: torch.Tensor
    regions: tuple[tuple[PartitionRegion, ...], ...]
    graph_diagnostics: tuple[dict[str, object], ...] = ()
    budgets: dict[str, list[int]] | None = None


def load_recursion_policy(
    path: str | Path,
    device: torch.device,
) -> RecursionPolicy:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    policy = RecursionPolicy(
        feature_size=int(
            payload.get(
                "feature_size",
                len(POLICY_INPUT_FEATURE_NAMES),
            )
        )
    )
    policy.load_state_dict(payload["model"])
    return policy.to(device).eval()


def load_graph_adapter(
    path: str | Path,
    device: torch.device,
    *,
    blend_scale: float = 1.0,
) -> RecursiveSemanticGraphAdapter:
    if not 0.0 <= float(blend_scale) <= 1.0:
        raise ValueError("blend_scale must be in [0,1]")
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    adapter = RecursiveSemanticGraphAdapter(
        int(payload["num_classes"]),
        hidden_size=int(config.get("hidden_size", 64)),
        heads=int(config.get("heads", 4)),
        router_count=int(config.get("router_count", 8)),
        top_k=int(config.get("top_k", 4)),
        blend=float(config.get("blend", 0.1)),
    )
    adapter.load_state_dict(payload["model"])
    adapter.blend *= float(blend_scale)
    return adapter.to(device).eval()


def _context_crop(
    footprint: torch.Tensor,
    bbox: tuple[int, int, int, int],
    context_fraction: float,
) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int, int, int]]:
    top, left, bottom, right = bbox
    height, width = footprint.shape[-2:]
    pad_y = round((bottom - top) * float(context_fraction))
    pad_x = round((right - left) * float(context_fraction))
    context = (
        max(0, top - pad_y),
        max(0, left - pad_x),
        min(height, bottom + pad_y),
        min(width, right + pad_x),
    )
    ctop, cleft, cbottom, cright = context
    return (
        footprint[:, ctop:cbottom, cleft:cright],
        context,
        (top - ctop, left - cleft, bottom - ctop, right - cleft),
    )


def decode_partition_region_logits(
    segmenter: ContourRoomSegmenter,
    footprint: torch.Tensor,
    regions: tuple[PartitionRegion, ...] | list[PartitionRegion],
    *,
    global_logits: torch.Tensor | None = None,
    context_fraction: float = 0.25,
    image_size: int = 256,
    recursive_blend: float = 0.25,
) -> list[torch.Tensor]:
    """Decode contour crops once for inference or graph-adapter training."""

    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    if footprint.shape[0] != 1:
        raise ValueError("decode_partition_region_logits accepts one case")
    decoded = []
    for region in regions:
        crop, context, local_bbox = _context_crop(
            footprint,
            region.bbox,
            context_fraction,
        )
        model_input = F.interpolate(
            crop[None].float(),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        context_logits = segmenter(model_input)[0]
        context_logits = F.interpolate(
            context_logits[None].float(),
            size=(
                context[2] - context[0],
                context[3] - context[1],
            ),
            mode="bilinear",
            align_corners=False,
        )[0]
        top, left, bottom, right = local_bbox
        local = context_logits[:, top:bottom, left:right]
        if global_logits is not None:
            anchor_top, anchor_left, anchor_bottom, anchor_right = region.bbox
            anchor = global_logits[
                0,
                :,
                anchor_top:anchor_bottom,
                anchor_left:anchor_right,
            ].float()
            local = (
                (1.0 - float(recursive_blend)) * anchor
                + float(recursive_blend) * local.float()
            )
        decoded.append(local)
    return decoded


@torch.inference_mode()
def predict_recursive_room_prior(
    policy: RecursionPolicy,
    segmenter: ContourRoomSegmenter,
    footprint: torch.Tensor,
    *,
    max_depth: int = 3,
    min_leaf_pixels: int = 256,
    context_fraction: float = 0.25,
    image_size: int = 256,
    global_logits: torch.Tensor | None = None,
    recursive_blend: float = 0.25,
    graph_blend: float = 0.0,
    graph_steps: int = 2,
    graph_margin_threshold: float | None = None,
    room_group_index: torch.Tensor | None = None,
    graph_adapter: RecursiveSemanticGraphAdapter | None = None,
) -> RecursiveRoomPriorOutput:
    """Decode direct or recursive room logits using footprint input only."""

    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    device = footprint.device
    if not 0.0 <= float(recursive_blend) <= 1.0:
        raise ValueError("recursive_blend must be in [0, 1]")
    features = complexity_features(footprint, None, None)
    features = features[:, : len(POLICY_INPUT_FEATURE_NAMES)]
    decision = policy.decide(features.to(device))
    budgets = allocate_dynamic_budgets(
        decision["recursion_strength"],
        max_depth=max_depth,
    )
    all_logits = []
    all_regions = []
    all_graph_diagnostics = []
    for index in range(footprint.shape[0]):
        regions = propose_partition(
            footprint[index].detach().cpu(),
            float(decision["recursion_strength"][index]),
            max_depth=min(
                int(max_depth),
                int(budgets.recursion_depth[index]),
            ),
            min_leaf_pixels=min_leaf_pixels,
        )
        local_logits = decode_partition_region_logits(
            segmenter,
            footprint[index : index + 1],
            regions,
            global_logits=(
                None
                if global_logits is None
                else global_logits[index : index + 1]
            ),
            context_fraction=context_fraction,
            image_size=image_size,
            recursive_blend=recursive_blend,
        )
        graph = build_partition_graph(regions)
        if graph_adapter is not None:
            local_logits, learned_diagnostics = graph_adapter(
                local_logits,
                graph,
                recursion_strength=float(
                    decision["recursion_strength"][index]
                ),
            )
            graph_diagnostics = learned_diagnostics
        case_graph_steps = min(
            int(graph_steps),
            int(budgets.graph_steps[index]),
        )
        if graph_blend > 0 and case_graph_steps > 0:
            local_logits, message_diagnostics = graph_refine_region_logits(
                local_logits,
                graph,
                blend=graph_blend,
                steps=case_graph_steps,
                margin_threshold=graph_margin_threshold,
                room_group_index=room_group_index,
            )
            if graph_adapter is not None:
                graph_diagnostics = {
                    **graph_diagnostics,
                    **message_diagnostics,
                }
            else:
                graph_diagnostics = message_diagnostics
        elif graph_adapter is None:
            graph_diagnostics = graph.to_dict()
        if graph_adapter is not None:
            graph_diagnostics = {
                **graph_diagnostics,
                "query_budget": int(budgets.query_budget[index]),
                "expert_budget": int(budgets.expert_budget[index]),
                "recursion_depth": int(budgets.recursion_depth[index]),
                "graph_steps": case_graph_steps,
            }
        all_graph_diagnostics.append(graph_diagnostics)
        all_logits.append(
            merge_region_logits(
                footprint[index].detach().cpu(),
                regions,
                [value.cpu() for value in local_logits],
            ).to(device)
        )
        all_regions.append(tuple(regions))
    return RecursiveRoomPriorOutput(
        logits=torch.stack(all_logits),
        action=decision["action"],
        recursion_strength=decision["recursion_strength"],
        regions=tuple(all_regions),
        graph_diagnostics=tuple(all_graph_diagnostics),
        budgets=budgets.to_dict(),
    )


__all__ = [
    "POLICY_ACTION_NAMES",
    "RecursiveRoomPriorOutput",
    "decode_partition_region_logits",
    "load_graph_adapter",
    "load_recursion_policy",
    "predict_recursive_room_prior",
]

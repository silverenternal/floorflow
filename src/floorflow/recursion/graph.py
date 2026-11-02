"""Spatial partition graph and dynamic room-type message passing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from floorflow.recursion.partition import PartitionRegion


@dataclass(frozen=True)
class PartitionGraph:
    regions: tuple[PartitionRegion, ...]
    adjacency: torch.Tensor
    degrees: torch.Tensor

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": len(self.regions),
            "edges": int((self.adjacency > 0).sum() // 2),
            "degrees": self.degrees.tolist(),
        }


def build_partition_graph(regions: Sequence[PartitionRegion]) -> PartitionGraph:
    count = len(regions)
    adjacency = np.eye(count, dtype=np.float32)
    structure = np.ones((3, 3), dtype=bool)
    masks = [region.mask.detach().cpu().numpy().astype(bool) for region in regions]
    for left in range(count):
        near = binary_dilation(masks[left], structure=structure)
        for right in range(left + 1, count):
            if np.logical_and(near, masks[right]).any():
                adjacency[left, right] = adjacency[right, left] = 1.0
    adjacency_tensor = torch.from_numpy(adjacency)
    adjacency_tensor /= adjacency_tensor.sum(dim=-1, keepdim=True).clamp_min(1)
    return PartitionGraph(
        regions=tuple(regions),
        adjacency=adjacency_tensor,
        degrees=torch.from_numpy((adjacency > 0).sum(axis=1) - 1).long(),
    )


def graph_refine_region_logits(
    local_logits: Sequence[torch.Tensor],
    graph: PartitionGraph,
    *,
    blend: float = 0.1,
    steps: int = 2,
    margin_threshold: float | None = None,
    room_group_index: torch.Tensor | None = None,
    preserve_group_argmax: bool = True,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    """Propagate room-type evidence through adjacent partition nodes."""

    if not 0.0 <= float(blend) <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    if len(local_logits) != len(graph.regions):
        raise ValueError("one local logit tensor is required per graph node")
    scores = []
    for logits, region in zip(local_logits, graph.regions):
        top, left, bottom, right = region.bbox
        local_mask = region.mask[top:bottom, left:right].to(logits.device)
        if int(local_mask.sum()) == 0:
            scores.append(logits.mean(dim=(1, 2)))
            continue
        selected = logits[:, local_mask]
        scores.append(selected.mean(dim=1))
    original = torch.stack(scores)
    probability = original.float().log_softmax(dim=-1).exp()
    if room_group_index is not None:
        group_index = room_group_index.to(
            device=probability.device,
            dtype=torch.long,
        )
        if group_index.shape != (probability.shape[-1],):
            raise ValueError("room_group_index must match class count")
        group_count = int(group_index.max()) + 1
        group_probability = torch.zeros(
            (probability.shape[0], group_count),
            device=probability.device,
            dtype=probability.dtype,
        )
        group_probability.scatter_add_(
            1,
            group_index[None].expand(probability.shape[0], -1),
            probability,
        )
        original_group_argmax = group_probability.argmax(dim=-1)
    else:
        group_index = None
        group_probability = None
        original_group_argmax = None
    update_weight = torch.full(
        (probability.shape[0], 1),
        float(blend),
        device=probability.device,
        dtype=probability.dtype,
    )
    if margin_threshold is not None:
        top_two = probability.topk(2, dim=-1).values
        margin = top_two[:, 0] - top_two[:, 1]
        update_weight = update_weight * (
            margin <= float(margin_threshold)
        )[:, None].to(update_weight.dtype)
    adjacency = graph.adjacency.to(
        device=probability.device,
        dtype=probability.dtype,
    )
    refined = probability
    for _ in range(max(0, int(steps))):
        if group_probability is not None and group_index is not None:
            refined = torch.zeros_like(probability)
            for group in range(group_probability.shape[1]):
                selected = group_index == group
                if not selected.any():
                    continue
                segment = probability[:, selected]
                mass = segment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                smooth = (1.0 - update_weight) * segment + (
                    update_weight * (adjacency @ segment)
                )
                smooth = smooth / smooth.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)
                refined[:, selected] = smooth * mass
        else:
            refined = (1.0 - update_weight) * refined + update_weight * (
                adjacency @ refined
            )
        refined = refined / refined.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if (
        preserve_group_argmax
        and group_index is not None
        and group_probability is not None
        and original_group_argmax is not None
    ):
        final_group_probability = torch.zeros_like(group_probability)
        final_group_probability.scatter_add_(
            1,
            group_index[None].expand(refined.shape[0], -1),
            refined,
        )
        wrong = (
            final_group_probability.argmax(dim=-1)
            != original_group_argmax
        )
        if wrong.any():
            rows = wrong.nonzero(as_tuple=False).flatten()
            for row in rows.tolist():
                target_group = int(original_group_argmax[row])
                other_max = torch.cat(
                    [
                        final_group_probability[row, :target_group],
                        final_group_probability[row, target_group + 1 :],
                    ]
                ).max()
                selected = group_index == target_group
                mass = refined[row, selected].sum().clamp_min(1e-8)
                scale = (other_max + 1.0e-5) / mass
                refined[row, selected] *= scale
            refined = refined / refined.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    bias = (
        refined.clamp_min(1e-8).log()
        - probability.clamp_min(1e-8).log()
    ).to(original.dtype)
    adjusted = [
        logits + bias[index, :, None, None]
        for index, logits in enumerate(local_logits)
    ]
    changed = (bias != 0).float().mean()
    return adjusted, {
        "nodes": len(graph.regions),
        "edges": int((graph.adjacency > 0).sum() // 2),
        "changed_fraction": float(changed),
        "mean_degree": float(graph.degrees.float().mean())
        if len(graph.regions)
        else 0.0,
        "updated_nodes": int((update_weight > 0).sum()),
        "margin_threshold": margin_threshold,
        "group_constrained": room_group_index is not None,
        "preserve_group_argmax": preserve_group_argmax,
    }


__all__ = [
    "PartitionGraph",
    "build_partition_graph",
    "graph_refine_region_logits",
]

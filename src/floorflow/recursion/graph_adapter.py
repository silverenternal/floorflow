"""Trainable semantic attention over recursive partition graph nodes.

The attention primitive is adapted from the project's pinned MiTA route
implementation.  This adapter turns coarse-to-fine region evidence into a
residual room-logit correction while preserving the existing segmenter.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from floorflow.model.official_mita_route import OfficialMiTARouteAttention
from floorflow.recursion.graph import PartitionGraph


def region_node_features(
    local_logits: Sequence[torch.Tensor],
    graph: PartitionGraph,
) -> torch.Tensor:
    """Pool per-region evidence and append topology-aware spatial descriptors."""

    if not local_logits:
        raise ValueError("at least one region is required")
    if len(local_logits) != len(graph.regions):
        raise ValueError("one logit tensor is required per graph node")
    device = local_logits[0].device
    dtype = local_logits[0].dtype
    pooled = []
    descriptors = []
    for logits, region in zip(local_logits, graph.regions):
        if logits.ndim != 3:
            raise ValueError("local logits must be [C,H,W]")
        top, left, bottom, right = region.bbox
        mask = region.mask[top:bottom, left:right].to(device)
        selected = logits[:, mask]
        if selected.shape[1] == 0:
            merged = logits.mean(dim=(1, 2))
        else:
            merged = torch.cat(
                [selected.mean(dim=1), selected.amax(dim=1)],
                dim=0,
            )
        pooled.append(merged)
        height = max(1, bottom - top)
        width = max(1, right - left)
        area = max(1.0, float(region.mask.sum()))
        center_y = (top + bottom) * 0.5 / max(1, graph.regions[0].mask.shape[0])
        center_x = (left + right) * 0.5 / max(1, graph.regions[0].mask.shape[1])
        descriptors.append(
            logits.new_tensor(
                [
                    float(region.depth) / 4.0,
                    area / float(region.mask.numel()),
                    math.log1p(height / width),
                    center_x,
                    center_y,
                    width / max(1, graph.regions[0].mask.shape[1]),
                    height / max(1, graph.regions[0].mask.shape[0]),
                    float(region.parent_index >= 0),
                ],
                dtype=dtype,
            )
        )
    return torch.cat(
        [torch.stack(pooled, dim=0), torch.stack(descriptors, dim=0)],
        dim=-1,
    )[None]


class RecursiveSemanticGraphAdapter(nn.Module):
    """Predict room-type corrections from routed parent/sibling evidence."""

    def __init__(
        self,
        num_classes: int,
        *,
        hidden_size: int = 64,
        heads: int = 4,
        router_count: int = 8,
        top_k: int = 4,
        blend: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or hidden_size <= 0:
            raise ValueError("num_classes and hidden_size must be positive")
        if not 0.0 <= float(blend) <= 1.0:
            raise ValueError("blend must be in [0,1]")
        self.num_classes = int(num_classes)
        self.hidden_size = int(hidden_size)
        self.router_count = int(router_count)
        self.top_k = int(top_k)
        self.blend = float(blend)
        self.input_projection = nn.Sequential(
            nn.Linear(self.num_classes * 2 + 8, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU(),
        )
        self.topology_embedding = nn.Linear(7, self.hidden_size, bias=False)
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.attention = OfficialMiTARouteAttention(
            self.hidden_size,
            num_heads=heads,
            router_count=self.router_count,
            top_k=self.top_k,
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.class_head = nn.Linear(self.hidden_size, self.num_classes)
        nn.init.zeros_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)

    @staticmethod
    def topology_features(graph: PartitionGraph) -> torch.Tensor:
        adjacency = graph.adjacency.float()
        degree = adjacency.sum(dim=-1, keepdim=True)
        second_moment = (adjacency * adjacency).sum(
            dim=-1,
            keepdim=True,
        )
        parents = torch.tensor(
            [region.parent_index for region in graph.regions],
            dtype=adjacency.dtype,
        )[:, None]
        depths = torch.tensor(
            [region.depth for region in graph.regions],
            dtype=adjacency.dtype,
        )[:, None]
        internal = {region.parent_index for region in graph.regions}
        leaves = torch.tensor(
            [region.index not in internal for region in graph.regions],
            dtype=adjacency.dtype,
        )
        topology = torch.cat(
            [
                degree,
                degree.square(),
                second_moment,
                (parents >= 0).to(adjacency.dtype),
                depths / max(1, max(region.depth for region in graph.regions)),
                adjacency.mean(dim=-1, keepdim=True),
                leaves[:, None],
            ],
            dim=-1,
        )
        return topology[..., :7]

    def forward_features(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        topology: torch.Tensor,
        *,
        recursion_strength: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if features.ndim != 3:
            raise ValueError("node features must be [B,N,D]")
        if features.shape[-1] != self.num_classes * 2 + 8:
            raise ValueError("class count differs from adapter configuration")
        tokens = self.input_projection(features)
        topology = topology.to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        if topology.shape != (features.shape[0], features.shape[1], 7):
            raise ValueError("topology features must be [B,N,7]")
        tokens = self.input_norm(
            tokens + self.topology_embedding(topology)
        )
        node_count = tokens.shape[1]
        router_count = min(self.router_count, node_count)
        router_budget = max(
            1,
            round(
                (1.0 + min(1.0, max(0.0, float(recursion_strength))))
                * math.sqrt(node_count)
            ),
        )
        router_budgets = torch.full(
            (1, router_count),
            min(router_budget, self.top_k, node_count),
            device=tokens.device,
            dtype=torch.long,
        )
        attended, diagnostics = self.attention(
            tokens,
            tokens,
            router_budgets=router_budgets,
        )
        messages = torch.einsum(
            "bij,bjd->bid",
            adjacency.to(
                device=tokens.device,
                dtype=tokens.dtype,
            ),
            attended,
        )
        refined = self.output_norm(tokens + attended + messages)
        correction = self.class_head(refined) * self.blend
        diagnostics = {
            "nodes": node_count,
            "edges": int((adjacency > 0).sum() // 2),
            "changed_fraction": float((correction != 0).float().mean()),
            "routing_entropy": float(
                diagnostics["routing_entropy"].detach()
            ),
            "attention_entropy": float(
                diagnostics["attention_entropy"].detach()
            ),
            "effective_expert_fraction": float(
                diagnostics["effective_expert_fraction"].detach()
            ),
            "router_budget_mean": float(diagnostics["top_k"].detach()),
        }
        return correction, diagnostics

    def forward(
        self,
        local_logits: Sequence[torch.Tensor],
        graph: PartitionGraph,
        *,
        recursion_strength: float = 1.0,
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        features = region_node_features(local_logits, graph).to(
            device=local_logits[0].device,
            dtype=local_logits[0].dtype,
        )
        if features.shape[1] != len(graph.regions):
            raise ValueError("node features and graph nodes disagree")
        correction, diagnostics = self.forward_features(
            features,
            graph.adjacency[None],
            self.topology_features(graph)[None],
            recursion_strength=recursion_strength,
        )
        adjusted = [
            logits + correction[0, index, :, None, None].to(logits.dtype)
            for index, logits in enumerate(local_logits)
        ]
        return adjusted, diagnostics


__all__ = [
    "RecursiveSemanticGraphAdapter",
    "region_node_features",
]

"""Coarse semantic room program and graph-conditioned partition refinement."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.geometry.boundary_geometry import BoundaryGeometryEncoder
from floorflow.heads.room_program import RoomProgramHead
from floorflow.heads.room_region_refiner import RoomRegionRefiner


class CoarseRoomRegionHead(nn.Module):
    """Predict a low-resolution semantic region map from condition features."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_dim: int = 96,
        prior_channels: int = 0,
    ) -> None:
        super().__init__()
        self.prior_channels = int(prior_channels)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
            nn.SiLU(),
        )
        self.class_embedding = nn.Parameter(
            torch.randn(1, num_room_classes, hidden_dim) * 0.02
        )
        self.output = nn.Conv2d(hidden_dim, num_room_classes, 1)
        self.prior = (
            nn.Conv2d(prior_channels, num_room_classes, 1)
            if prior_channels
            else None
        )
        for module in (self.output, self.prior):
            if module is not None:
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,
        resolution: int,
        prior: torch.Tensor | None = None,
        class_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = F.adaptive_avg_pool2d(features, (resolution, resolution))
        hidden = self.project(hidden)
        class_maps = self.output(hidden)
        class_queries = self.class_embedding.to(hidden.dtype)
        if class_tokens is not None:
            class_queries = class_queries + class_tokens.to(hidden.dtype)
        class_maps = class_maps + torch.einsum(
            "bdhw,bkd->bkhw",
            hidden,
            class_queries,
        ) / math.sqrt(hidden.shape[1])
        if self.prior is not None:
            if prior is None:
                prior = torch.zeros(
                    features.shape[0],
                    self.prior_channels,
                    resolution,
                    resolution,
                    device=features.device,
                    dtype=features.dtype,
                )
            prior = F.adaptive_avg_pool2d(prior.to(hidden.dtype), (resolution, resolution))
            class_maps = class_maps + self.prior(prior)
        return class_maps


class SemanticRoomGraphReasoner(nn.Module):
    """Reason over room nodes with spatial co-occurrence adjacency."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.num_room_classes = int(num_room_classes)
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.descriptor = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.spatial_projection = nn.Conv2d(in_channels, hidden_dim, 1)
        self.output_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def adjacency_from_regions(probabilities: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool2d(probabilities, 3, stride=1, padding=1)
        adjacency = torch.einsum("bihw,bjhw->bij", probabilities, pooled)
        adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
        adjacency = adjacency / adjacency.amax(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
        eye = torch.eye(
            probabilities.shape[1],
            device=probabilities.device,
            dtype=probabilities.dtype,
        )[None]
        adjacency = (adjacency + eye).clamp_min(0.0)
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

    @staticmethod
    def region_descriptors(probabilities: torch.Tensor) -> torch.Tensor:
        _batch, _classes, height, width = probabilities.shape
        mass = probabilities.sum(dim=(2, 3))
        ys = torch.linspace(0.0, 1.0, height, device=probabilities.device, dtype=probabilities.dtype)
        xs = torch.linspace(0.0, 1.0, width, device=probabilities.device, dtype=probabilities.dtype)
        centroid_y = torch.einsum(
            "bkhw,h->bk",
            probabilities,
            ys,
        ) / mass.clamp_min(1.0e-6)
        centroid_x = torch.einsum(
            "bkhw,w->bk",
            probabilities,
            xs,
        ) / mass.clamp_min(1.0e-6)
        area = mass / float(height * width)
        return torch.stack(
            [
                area,
                centroid_x,
                centroid_y,
                area.sqrt(),
                torch.ones_like(area),
            ],
            dim=-1,
        )

    def forward(
        self,
        features: torch.Tensor,
        coarse_logits: torch.Tensor,
        class_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        probabilities = coarse_logits.softmax(dim=1)
        adjacency = self.adjacency_from_regions(probabilities)
        descriptors = self.region_descriptors(probabilities)
        nodes = (
            class_tokens
            + self.descriptor(descriptors)
            + self.global_context(class_tokens.mean(dim=1, keepdim=True))
        )
        messages = torch.einsum("bij,bjd->bid", adjacency, nodes)
        refined = self.norm(nodes + self.message(torch.cat([nodes, messages], dim=-1)))
        spatial = self.spatial_projection(features)
        semantic_bias = torch.einsum("bkd,bdhw->bkhw", refined, spatial)
        semantic_bias = semantic_bias / math.sqrt(spatial.shape[1])
        return {
            "graph_tokens": refined,
            "adjacency": adjacency,
            "region_descriptors": descriptors,
            "semantic_bias": semantic_bias * self.output_scale.to(semantic_bias.dtype),
        }


class RoomQueryGraphRefiner(nn.Module):
    """Refine room-query geometry through adjacency inferred from predicted masks."""

    def __init__(self, hidden_dim: int = 96) -> None:
        super().__init__()
        self.descriptor = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.graph_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        queries: torch.Tensor,
        mask_logits: torch.Tensor,
        footprint: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if queries.ndim != 3 or mask_logits.ndim != 4:
            raise ValueError("queries and mask_logits must be [B,N,H] and [B,N,H,W]")
        if queries.shape[:2] != mask_logits.shape[:2]:
            raise ValueError("query and mask cardinalities must match")
        probabilities = mask_logits.sigmoid()
        if footprint is not None:
            support = F.interpolate(
                footprint.float(),
                size=probabilities.shape[-2:],
                mode="nearest",
            )
            if support.shape[1] != 1:
                raise ValueError("footprint must have one channel")
            probabilities = probabilities * support
        adjacency = SemanticRoomGraphReasoner.adjacency_from_regions(
            probabilities
        )
        descriptors = SemanticRoomGraphReasoner.region_descriptors(
            probabilities
        )
        nodes = queries + self.descriptor(descriptors)
        messages = torch.einsum("bij,bjh->bih", adjacency, nodes)
        refined = self.norm(
            queries
            + torch.tanh(self.graph_scale.to(queries.dtype))
            * self.message(torch.cat([nodes, messages], dim=-1))
        )
        return {
            "queries": refined,
            "adjacency": adjacency,
            "descriptors": descriptors,
        }


class HierarchicalRoomPipeline(nn.Module):
    """Semantic room program followed by graph-conditioned region refinement."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_dim: int = 96,
        tokens: int = 8,
        coarse_resolution: int = 64,
        prior_channels: int = 0,
    ) -> None:
        super().__init__()
        self.coarse_resolution = int(coarse_resolution)
        self.program = RoomProgramHead(
            in_channels,
            num_room_classes,
            hidden_dim=hidden_dim,
            tokens=tokens,
        )
        self.boundary = BoundaryGeometryEncoder(hidden_dim=hidden_dim)
        self.boundary_attention = nn.MultiheadAttention(
            hidden_dim,
            4,
            batch_first=True,
        )
        self.boundary_norm = nn.LayerNorm(hidden_dim)
        self.coarse_regions = CoarseRoomRegionHead(
            in_channels,
            num_room_classes,
            hidden_dim=hidden_dim,
            prior_channels=prior_channels,
        )
        self.graph = SemanticRoomGraphReasoner(
            in_channels,
            num_room_classes,
            hidden_dim=hidden_dim,
        )
        self.refiner = RoomRegionRefiner(
            in_channels,
            num_room_classes,
            hidden_dim=hidden_dim,
        )
        self.boundary_enabled = True

    def forward(
        self,
        features: torch.Tensor,
        prior_logits: torch.Tensor | None = None,
        footprint: torch.Tensor | None = None,
        layout_prior_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        program = self.program(features)
        boundary = None
        class_tokens = program["class_tokens"]
        if footprint is not None and self.boundary_enabled:
            boundary = self.boundary(footprint)
            key_valid = boundary["valid"].clone()
            empty = ~key_valid.any(dim=1)
            key_valid[empty, 0] = True
            attended, _weights = self.boundary_attention(
                class_tokens,
                boundary["tokens"],
                boundary["tokens"],
                key_padding_mask=~key_valid,
                need_weights=False,
            )
            class_tokens = self.boundary_norm(class_tokens + attended)
            program = dict(program)
            program["class_tokens"] = class_tokens
        coarse_logits = self.coarse_regions(
            features,
            self.coarse_resolution,
            prior=(
                prior_logits
                if layout_prior_logits is None
                else layout_prior_logits
            ),
            class_tokens=class_tokens,
        )
        graph = self.graph(
            features,
            coarse_logits,
            class_tokens,
        )
        refined = self.refiner(
            features,
            coarse_logits,
            graph["graph_tokens"],
        )
        return {
            "program": program,
            "coarse_room_logits": coarse_logits,
            "coarse_room_probabilities": coarse_logits.softmax(dim=1),
            "refined_room_logits": refined["refined_room_logits"],
            "refined_room_probabilities": refined["refined_room_probabilities"],
            "room_region_attention": refined["room_region_attention"],
            "semantic_bias": graph["semantic_bias"],
            "room_graph_tokens": graph["graph_tokens"],
            "room_adjacency": graph["adjacency"],
            "room_region_descriptors": graph["region_descriptors"],
            "boundary_geometry": boundary,
        }

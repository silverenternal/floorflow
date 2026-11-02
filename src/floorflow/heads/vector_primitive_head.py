"""Query-based vector primitive prediction for rooms, walls, and openings."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.heads.room_adjacency import room_adjacency_logits


def _stage_summary(
    tokens: torch.Tensor,
    logits: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = (
        tokens.new_ones(tokens.shape[:2])
        if logits is None
        else logits.float().sigmoid()
    )
    if valid is not None:
        weights = weights.masked_fill(~valid, 0.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    return torch.einsum("bn,bnd->bd", weights, tokens)


def _anchor_grid(count: int) -> torch.Tensor:
    side = max(1, math.ceil(math.sqrt(count)))
    values = torch.linspace(0.05, 0.95, side)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    anchors = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return anchors[:count]


def _query_refiner(
    hidden_dim: int,
    heads: int,
    layers: int,
) -> nn.Module:
    return nn.ModuleList(
        [
            nn.TransformerDecoderLayer(
                hidden_dim,
                heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(max(1, int(layers)))
        ]
    )


class VectorPrimitiveHead(nn.Module):
    """Decode variable-length vector primitives from a spatial feature map."""

    def __init__(
        self,
        in_channels: int,
        room_classes: int,
        opening_classes: int,
        hidden_dim: int = 128,
        max_rooms: int = 32,
        max_walls: int = 64,
        max_openings: int = 48,
        polygon_points: int = 16,
        decoder_layers: int = 2,
        attention_heads: int = 4,
        stage_context: bool = False,
        ordered_stage_context: bool = False,
        stage_context_scale: float = 0.25,
        host_conditioned_openings: bool = False,
        point_validity: bool = False,
        partition_edge_condition: bool = False,
    ) -> None:
        super().__init__()
        self.max_rooms = int(max_rooms)
        self.max_walls = int(max_walls)
        self.max_openings = int(max_openings)
        self.polygon_points = int(polygon_points)
        self.stage_context = bool(stage_context)
        self.ordered_stage_context = bool(ordered_stage_context)
        self.stage_context_scale = float(stage_context_scale)
        if self.stage_context_scale < 0.0:
            raise ValueError("stage context scale must be non-negative")
        self.host_conditioned_openings = bool(host_conditioned_openings)
        self.partition_edge_condition = bool(partition_edge_condition)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
            nn.SiLU(),
        )
        self.level_embedding = nn.Parameter(
            torch.randn(1, hidden_dim, 4) * 0.02
        )
        self.room_queries = nn.Parameter(
            torch.randn(1, max_rooms, hidden_dim) * 0.02
        )
        self.wall_queries = nn.Parameter(
            torch.randn(1, max_walls, hidden_dim) * 0.02
        )
        self.opening_queries = nn.Parameter(
            torch.randn(1, max_openings, hidden_dim) * 0.02
        )
        self.register_buffer(
            "room_anchors",
            _anchor_grid(max_rooms),
            persistent=False,
        )
        self.register_buffer(
            "wall_anchors",
            _anchor_grid(max_walls),
            persistent=False,
        )
        self.register_buffer(
            "opening_anchors",
            _anchor_grid(max_openings),
            persistent=False,
        )
        self.anchor_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoders = nn.ModuleDict(
            {
                name: _query_refiner(
                    hidden_dim,
                    attention_heads,
                    decoder_layers,
                )
                for name in ("room", "wall", "opening")
            }
        )
        self.room_exists = nn.Linear(hidden_dim, 1)
        self.room_class = nn.Linear(hidden_dim, room_classes)
        self.room_adjacency_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        nn.init.eye_(self.room_adjacency_projection.weight)
        self.room_polygon = nn.Linear(
            hidden_dim,
            polygon_points * 2,
        )
        self.room_point_valid = (
            nn.Linear(hidden_dim, polygon_points)
            if point_validity
            else None
        )
        self.wall_exists = nn.Linear(hidden_dim, 1)
        self.wall_endpoints = nn.Linear(hidden_dim, 4)
        self.wall_thickness = nn.Linear(hidden_dim, 1)
        self.wall_exterior = nn.Linear(hidden_dim, 1)
        self.opening_exists = nn.Linear(hidden_dim, 1)
        self.opening_center = nn.Linear(hidden_dim, 2)
        self.opening_width = nn.Linear(hidden_dim, 1)
        self.opening_class = nn.Linear(hidden_dim, opening_classes)
        self.opening_host = (
            nn.Linear(hidden_dim, max_walls)
            if self.host_conditioned_openings
            else None
        )
        self.partition_edge_projection = None
        if self.partition_edge_condition:
            with torch.random.fork_rng(devices=[]):
                self.partition_edge_projection = nn.Conv2d(
                    1,
                    hidden_dim,
                    3,
                    padding=1,
                )
            nn.init.zeros_(self.partition_edge_projection.weight)
            nn.init.zeros_(self.partition_edge_projection.bias)
        self.stage_conditioners = (
            nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim * 2),
                    )
                    for name in ("wall", "opening")
                }
            )
            if self.ordered_stage_context
            else None
        )
        if self.stage_conditioners is not None:
            for conditioner in self.stage_conditioners.values():
                nn.init.zeros_(conditioner[-1].weight)
                nn.init.zeros_(conditioner[-1].bias)

    def _memory(
        self,
        features: torch.Tensor,
        graph_tokens: torch.Tensor | None,
        graph_mask: torch.Tensor | None,
        partition_edge: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = F.adaptive_avg_pool2d(features, (64, 64))
        hidden = self.feature_projection(features)
        if self.partition_edge_projection is not None:
            if partition_edge is None:
                raise ValueError(
                    "partition edge condition requires an evidence map"
                )
            edge = F.interpolate(
                partition_edge[:, None].to(hidden.dtype),
                size=hidden.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            hidden = hidden + self.partition_edge_projection(edge)
        batch, channels, height, width = hidden.shape
        levels = [
            F.adaptive_avg_pool2d(hidden, (height, width)),
            F.adaptive_avg_pool2d(hidden, (max(1, height // 2), max(1, width // 2))),
            F.adaptive_avg_pool2d(hidden, (max(1, height // 4), max(1, width // 4))),
            F.adaptive_avg_pool2d(hidden, (max(1, height // 8), max(1, width // 8))),
        ]
        tokens = []
        for index, level in enumerate(levels):
            value = level + self.level_embedding[..., index : index + 1].view(
                1,
                channels,
                1,
                1,
            )
            tokens.append(value.flatten(2).transpose(1, 2))
        memory = torch.cat(tokens, dim=1)
        if graph_tokens is not None:
            memory = torch.cat([memory, graph_tokens], dim=1)
            spatial_mask = torch.ones(
                batch,
                memory.shape[1] - graph_tokens.shape[1],
                dtype=torch.bool,
                device=memory.device,
            )
            if graph_mask is not None:
                graph_mask = graph_mask.bool()
            else:
                graph_mask = torch.ones(
                    batch,
                    graph_tokens.shape[1],
                    dtype=torch.bool,
                    device=memory.device,
                )
            valid = torch.cat([spatial_mask, graph_mask], dim=1)
            return memory, ~valid
        return memory, None

    def _decode(
        self,
        name: str,
        queries: torch.Tensor,
        anchors: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None,
        query_prior: torch.Tensor | None = None,
        stage_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchors.ndim == 2:
            anchors = anchors[None].expand(memory.shape[0], -1, -1)
        if anchors.shape[:2] != (memory.shape[0], queries.shape[1]):
            raise ValueError(f"{name} anchors must match batch and query count")
        hidden = (
            queries.expand(memory.shape[0], -1, -1)
            + self.anchor_projection(anchors)
        )
        if query_prior is not None:
            if query_prior.shape != hidden.shape:
                raise ValueError(
                    f"{name} query prior must match {tuple(hidden.shape)}"
                )
            hidden = hidden + query_prior.to(hidden.dtype)
        if stage_token is not None:
            if self.stage_conditioners is None:
                raise ValueError(
                    "stage tokens require ordered stage context"
                )
            if stage_token.ndim != 2:
                raise ValueError("stage token must be [B,C]")
            stage_token = stage_token[:, None].expand(
                -1,
                hidden.shape[1],
                -1,
            )
            scale, bias = self.stage_conditioners[name](
                stage_token
            ).chunk(2, dim=-1)
            hidden = (
                hidden
                * (1.0 + self.stage_context_scale * torch.tanh(scale))
                + self.stage_context_scale * torch.tanh(bias)
            )
        for layer in self.decoders[name]:
            hidden = layer(
                hidden,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return hidden

    def forward(
        self,
        features: torch.Tensor,
        graph_tokens: torch.Tensor | None = None,
        graph_mask: torch.Tensor | None = None,
        room_query_prior: torch.Tensor | None = None,
        room_query_anchors: torch.Tensor | None = None,
        room_polygon_prior: torch.Tensor | None = None,
        partition_edge: torch.Tensor | None = None,
        stage_tokens: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if (
            self.ordered_stage_context
            and stage_tokens is None
        ):
            raise ValueError("ordered stage context requires stage tokens")
        if (
            self.ordered_stage_context
            and not {"walls", "junctions", "openings"}.issubset(stage_tokens)
        ):
            raise ValueError(
                "ordered stage context requires walls, junctions, and openings"
            )
        memory, memory_mask = self._memory(
            features,
            graph_tokens,
            graph_mask,
            partition_edge,
        )
        room_anchors = (
            self.room_anchors[None].expand(memory.shape[0], -1, -1)
            if room_query_anchors is None
            else room_query_anchors.to(memory.dtype)
        )
        rooms = self._decode(
            "room",
            self.room_queries,
            room_anchors,
            memory,
            memory_mask,
            room_query_prior,
        )
        wall_memory = memory
        wall_mask = memory_mask
        if self.stage_context:
            room_active = self.room_exists(rooms).squeeze(-1).sigmoid() >= 0.5
            room_padding = ~room_active
            wall_memory = torch.cat([memory, rooms], dim=1)
            spatial_mask = (
                memory_mask
                if memory_mask is not None
                else torch.zeros(
                    memory.shape[0],
                    memory.shape[1],
                    dtype=torch.bool,
                    device=memory.device,
                )
            )
            wall_mask = torch.cat([spatial_mask, room_padding], dim=1)
        walls = self._decode(
            "wall",
            self.wall_queries,
            self.wall_anchors,
            wall_memory,
            wall_mask,
            stage_token=(
                stage_tokens["walls"] if stage_tokens is not None else None
            ),
        )
        opening_memory = wall_memory
        opening_mask = wall_mask
        if self.stage_context:
            wall_active = self.wall_exists(walls).squeeze(-1).sigmoid() >= 0.5
            wall_padding = ~wall_active
            opening_memory = torch.cat([wall_memory, walls], dim=1)
            opening_mask = torch.cat([wall_mask, wall_padding], dim=1)
        openings = self._decode(
            "opening",
            self.opening_queries,
            self.opening_anchors,
            opening_memory,
            opening_mask,
            stage_token=(
                0.5
                * (
                    stage_tokens["junctions"]
                    + stage_tokens["openings"]
                )
                if stage_tokens is not None
                else None
            ),
        )
        room_polygon_offsets = self.room_polygon(rooms).reshape(
            rooms.shape[0],
            self.max_rooms,
            self.polygon_points,
            2,
        )
        if room_polygon_prior is None:
            room_polygons = (
                room_anchors[:, :, None, :]
                + 0.5 * torch.tanh(room_polygon_offsets)
            )
        else:
            if room_polygon_prior.shape != room_polygon_offsets.shape:
                raise ValueError(
                    "room_polygon_prior must match [B,Q,P,2]"
                )
            room_polygons = (
                room_polygon_prior.to(room_polygon_offsets.dtype)
                + 0.25 * torch.tanh(room_polygon_offsets)
            )
        wall_endpoint_offsets = self.wall_endpoints(walls).reshape(
            walls.shape[0],
            self.max_walls,
            2,
            2,
        )
        adjacency_logits = room_adjacency_logits(
            rooms,
            self.room_adjacency_projection,
        )
        output = {
            "room_features": rooms,
            "room_exists": self.room_exists(rooms).squeeze(-1),
            "room_class": self.room_class(rooms),
            "room_adjacency_logits": adjacency_logits,
            "room_polygons": room_polygons,
            "wall_exists": self.wall_exists(walls).squeeze(-1),
            "wall_endpoints": (
                self.wall_anchors.view(
                    1,
                    self.max_walls,
                    1,
                    2,
                )
                + 0.5 * torch.tanh(wall_endpoint_offsets)
            ),
            "wall_thickness": self.wall_thickness(walls).squeeze(-1),
            "wall_exterior": self.wall_exterior(walls).squeeze(-1),
            "wall_exterior_logits": self.wall_exterior(walls).squeeze(-1),
            "opening_exists": self.opening_exists(openings).squeeze(-1),
            "opening_centers": (
                self.opening_anchors.view(
                    1,
                    self.max_openings,
                    2,
                )
                + 0.25 * torch.tanh(self.opening_center(openings))
            ),
            "opening_widths": self.opening_width(openings).squeeze(-1),
            "opening_class": self.opening_class(openings),
            "vector_stage_tokens": {
                "envelope": _stage_summary(
                    memory,
                    valid=(~memory_mask if memory_mask is not None else None),
                ),
                "rooms": _stage_summary(
                    rooms,
                    self.room_exists(rooms).squeeze(-1),
                ),
                "walls": _stage_summary(
                    walls,
                    self.wall_exists(walls).squeeze(-1),
                ),
                "openings": _stage_summary(
                    openings,
                    self.opening_exists(openings).squeeze(-1),
                ),
            },
        }
        if self.opening_host is not None:
            output["opening_host_logits"] = self.opening_host(openings)
        if self.room_point_valid is not None:
            output["room_point_valid_logits"] = self.room_point_valid(rooms)
        return output


__all__ = ["VectorPrimitiveHead"]

"""Bounded token-conditioned residual editing of the discrete CAD graph."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.objectives.token_grammar import TOKEN_STAGES


class TokenGraphResidualHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        room_classes: int,
        *,
        max_rooms: int,
        max_walls: int,
        max_openings: int,
        max_junctions: int = 32,
        degree_classes: int = 9,
        residual_scale: float = 0.5,
        min_confidence: float = 0.05,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.room_classes = int(room_classes)
        self.max_rooms = int(max_rooms)
        self.max_walls = int(max_walls)
        self.max_openings = int(max_openings)
        self.max_junctions = int(max_junctions)
        self.degree_classes = int(degree_classes)
        self.residual_scale = float(residual_scale)
        self.min_confidence = float(min_confidence)
        context_dim = hidden_dim * (1 + len(TOKEN_STAGES))
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.gate = nn.Linear(hidden_dim, 1)
        heads = max(1, min(4, hidden_dim // 8))
        self.entity_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            batch_first=True,
        )
        self.entity_norm = nn.LayerNorm(hidden_dim)
        self.room_query = nn.Parameter(torch.randn(self.max_rooms, hidden_dim) * 0.02)
        self.wall_query = nn.Parameter(torch.randn(self.max_walls, hidden_dim) * 0.02)
        self.opening_query = nn.Parameter(
            torch.randn(self.max_openings, hidden_dim) * 0.02
        )
        self.junction_query = nn.Parameter(
            torch.randn(self.max_junctions, hidden_dim) * 0.02
        )
        self.room_id_embedding = nn.Embedding(self.max_rooms, hidden_dim)
        self.wall_id_embedding = nn.Embedding(self.max_walls, hidden_dim)
        self.opening_id_embedding = nn.Embedding(self.max_openings, hidden_dim)
        self.junction_id_embedding = nn.Embedding(self.max_junctions, hidden_dim)
        junction_side = int(max_junctions**0.5) + 1
        junction_anchor = torch.stack(
            torch.meshgrid(
                torch.linspace(0.05, 0.95, junction_side),
                torch.linspace(0.05, 0.95, junction_side),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)[: self.max_junctions]
        if len(junction_anchor) < self.max_junctions:
            junction_anchor = torch.nn.functional.pad(
                junction_anchor,
                (0, 0, 0, self.max_junctions - len(junction_anchor)),
            )
        self.junction_anchor = nn.Parameter(junction_anchor, requires_grad=True)
        self.room_exists = nn.Linear(hidden_dim, 1)
        self.room_class = nn.Linear(hidden_dim, self.room_classes)
        self.wall_exists = nn.Linear(hidden_dim, 1)
        self.wall_type = nn.Linear(hidden_dim, 1)
        self.opening_exists = nn.Linear(hidden_dim, 1)
        self.opening_host = nn.Linear(hidden_dim, self.max_walls)
        self.adjacency_scale = nn.Linear(hidden_dim, hidden_dim)
        self.junction_coordinates = nn.Linear(hidden_dim, 2)
        self.junction_degree = nn.Linear(hidden_dim, self.degree_classes)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        token_hidden: torch.Tensor,
        token_mask: torch.Tensor,
        stage_ids: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        weights = token_mask[..., None].to(token_hidden.dtype)
        global_context = (token_hidden * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        stage_contexts = []
        for stage_id in range(len(TOKEN_STAGES)):
            selected = token_mask & (stage_ids == stage_id)
            selected_float = selected[..., None].to(token_hidden.dtype)
            stage_contexts.append(
                (token_hidden * selected_float).sum(dim=1)
                / selected_float.sum(dim=1).clamp_min(1.0)
            )
        context = self.context(torch.cat([global_context, *stage_contexts], dim=-1))
        gate = self.gate(context).sigmoid()
        if confidence is not None:
            gate = gate * confidence.clamp(0.0, 1.0)[:, None]
            gate = gate.masked_fill(confidence[:, None] < self.min_confidence, 0.0)
        scale = self.residual_scale

        def attend(query: torch.Tensor) -> torch.Tensor:
            expanded = (
                query[None].expand(token_hidden.shape[0], -1, -1)
                if query.ndim == 2
                else query
            )
            attended = self.entity_attention(
                expanded,
                token_hidden,
                token_hidden,
                key_padding_mask=~token_mask,
                need_weights=False,
            )[0]
            return self.entity_norm(expanded + attended)

        room_ids = predictions.get("room_instance_ids")
        if room_ids is None:
            room_ids = torch.arange(
                self.max_rooms,
                device=token_hidden.device,
            )[None].expand(token_hidden.shape[0], -1)
        room_ids = room_ids.long().clamp(0, self.max_rooms - 1)
        room_context = attend(
            self.room_query + self.room_id_embedding(room_ids)[0]
        )
        wall_context = attend(
            self.wall_query + self.wall_id_embedding.weight
        )
        opening_context = attend(
            self.opening_query + self.opening_id_embedding.weight
        )
        junction_context = attend(
            self.junction_query + self.junction_id_embedding.weight
        )

        def residual(
            layer: nn.Linear,
            features: torch.Tensor,
            shape: tuple[int, ...],
        ) -> torch.Tensor:
            gate_view = gate.reshape(
                token_hidden.shape[0],
                *([1] * len(shape)),
            )
            return gate_view * scale * torch.tanh(layer(features)).reshape(
                token_hidden.shape[0],
                *shape,
            )

        def gate_for(rank: int) -> torch.Tensor:
            return gate.reshape(
                token_hidden.shape[0],
                *([1] * rank),
            )

        room_exists = residual(
            self.room_exists,
            room_context,
            (self.max_rooms,),
        )
        room_class = gate_for(2) * scale * torch.tanh(
            self.room_class(room_context)
        )
        wall_exists = residual(
            self.wall_exists,
            wall_context,
            (self.max_walls,),
        )
        wall_type = residual(
            self.wall_type,
            wall_context,
            (self.max_walls,),
        )
        opening_exists = residual(
            self.opening_exists,
            opening_context,
            (self.max_openings,),
        )
        opening_host = gate_for(2) * scale * torch.tanh(
            self.opening_host(opening_context)
        )
        room_edge = self.adjacency_scale(room_context)
        adjacency = scale * gate_for(2) * torch.tanh(
            torch.einsum("bid,bjd->bij", room_edge, room_context)
            / max(1.0, room_context.shape[-1]) ** 0.5
        )
        adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
        junction_coordinates = residual(
            self.junction_coordinates,
            junction_context,
            (self.max_junctions, 2),
        )
        junction_degree = gate_for(2) * scale * torch.tanh(
            self.junction_degree(junction_context)
        )
        return {
            **predictions,
            "room_exists": predictions["room_exists"] + room_exists,
            "room_class": predictions["room_class"] + room_class,
            "wall_exists": predictions["wall_exists"] + wall_exists,
            "wall_exterior_logits": (
                predictions["wall_exterior_logits"] + wall_type
            ),
            "opening_exists": predictions["opening_exists"] + opening_exists,
            "opening_host_logits": (
                predictions["opening_host_logits"] + opening_host
            ),
            "room_adjacency_logits": (
                predictions["room_adjacency_logits"] + adjacency
            ),
            "token_junction_degree_logits": junction_degree,
            "token_junction_coordinates": (
                self.junction_anchor[None] + junction_coordinates
            ),
            "token_graph_gate": gate[:, 0],
            "token_graph_room_residual_norm": room_exists.norm(dim=1),
            "token_graph_wall_residual_norm": wall_exists.norm(dim=1),
            "token_graph_opening_residual_norm": opening_exists.norm(dim=1),
            "token_graph_adjacency_residual_norm": adjacency.flatten(1).norm(dim=1),
            "token_graph_junction_residual_norm": junction_degree.flatten(1).norm(
                dim=1
            ),
        }


__all__ = ["TokenGraphResidualHead"]

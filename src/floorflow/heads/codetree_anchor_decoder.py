"""Two-level room-token to polygon-anchor decoder."""

from __future__ import annotations

import math

import torch
from torch import nn

from floorflow.heads.vector_quantizer import VectorQuantizer
from floorflow.objectives.token_grammar import TOKEN_STAGES


def _room_anchor(max_rooms: int, polygon_points: int) -> torch.Tensor:
    side = max(1, math.ceil(math.sqrt(max_rooms)))
    centers = torch.stack(
        torch.meshgrid(
            torch.linspace(0.1, 0.9, side),
            torch.linspace(0.1, 0.9, side),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)[:max_rooms]
    angles = torch.linspace(0.0, 2.0 * math.pi, polygon_points + 1)[:-1]
    offsets = torch.stack([angles.cos(), angles.sin()], dim=-1) * 0.04
    return centers[:, None] + offsets[None]


class CodeTreeAnchorDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        room_classes: int,
        *,
        max_rooms: int,
        polygon_points: int,
        heads: int = 4,
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim % heads:
            heads = 1
        self.max_rooms = int(max_rooms)
        self.room_classes = int(room_classes)
        self.polygon_points = int(polygon_points)
        self.residual_scale = float(residual_scale)
        self.room_anchor = nn.Parameter(
            _room_anchor(self.max_rooms, self.polygon_points),
            requires_grad=True,
        )
        self.room_query_embedding = nn.Parameter(
            torch.randn(self.max_rooms, hidden_dim) * 0.02,
        )
        self.room_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            batch_first=True,
        )
        self.room_norm = nn.LayerNorm(hidden_dim)
        self.polygon_delta = nn.Linear(
            hidden_dim,
            self.polygon_points * 2,
        )
        self.anchor_gate = nn.Linear(hidden_dim, 1)
        self.room_codebook = VectorQuantizer(32, hidden_dim)
        self.polygon_projection = nn.Linear(hidden_dim, hidden_dim)
        self.polygon_codebook = VectorQuantizer(64, hidden_dim)
        self.class_embedding = nn.Embedding(self.room_classes, hidden_dim)
        self.box_projection = nn.Linear(4, hidden_dim)

    def forward(
        self,
        token_hidden: torch.Tensor,
        token_mask: torch.Tensor,
        stage_ids: torch.Tensor,
        predictions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        room_stage = TOKEN_STAGES.index("rooms")
        room_mask = token_mask & (stage_ids == room_stage)
        fallback_mask = room_mask.any(dim=1, keepdim=True)
        room_mask = torch.where(fallback_mask, room_mask, token_mask)
        room_queries = self.room_query_embedding[None].expand(
            token_hidden.shape[0],
            -1,
            -1,
        )
        attended = self.room_attention(
            room_queries,
            token_hidden,
            token_hidden,
            key_padding_mask=~room_mask,
            need_weights=False,
        )[0]
        semantic = 0.0
        if predictions is not None and "room_class" in predictions:
            class_probability = predictions["room_class"][..., :-1].softmax(dim=-1)
            semantic = class_probability @ self.class_embedding.weight
        if predictions is not None and "room_polygons" in predictions:
            boxes = torch.cat(
                [
                    predictions["room_polygons"].amin(dim=2),
                    predictions["room_polygons"].amax(dim=2),
                ],
                dim=-1,
            )
            semantic = semantic + self.box_projection(boxes)
        room_context = self.room_norm(room_queries + attended + semantic)
        room_code, room_indices, room_vq_loss = self.room_codebook(room_context)
        polygon_code, polygon_indices, polygon_vq_loss = self.polygon_codebook(
            self.polygon_projection(room_context)
        )
        decoder_context = room_code + polygon_code
        gate = self.anchor_gate(decoder_context).sigmoid()
        delta = self.polygon_delta(decoder_context).reshape(
            token_hidden.shape[0],
            self.max_rooms,
            self.polygon_points,
            2,
        )
        polygons = (
            self.room_anchor[None]
            + self.residual_scale * gate[..., None] * torch.tanh(delta)
        )
        return {
            "codetree_room_tokens": room_context,
            "codetree_polygons": polygons,
            "codetree_anchor_gate": gate[:, 0],
            "codetree_room_codes": room_indices,
            "codetree_polygon_codes": polygon_indices,
            "codetree_vq_loss": room_vq_loss + polygon_vq_loss,
            "codetree_polygon_residual_norm": (
                self.residual_scale * gate[..., None] * torch.tanh(delta)
            ).flatten(1).norm(dim=1),
        }


__all__ = ["CodeTreeAnchorDecoder"]

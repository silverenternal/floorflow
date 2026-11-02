"""Token-logit implied soft graph consistency with vector heads."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.objectives.token_grammar import FloorplanTokenVocabulary
from floorflow.representation.unified_floorplan_tokens import FloorplanToken

TOKEN_SOFT_GRAPH_VERSION = "floorflow-token-soft-graph-v1"


def _entity_index(value: str, prefix: str) -> int | None:
    if not value.startswith(prefix):
        return None
    try:
        return int(value.rsplit("_", 1)[-1]) - 1
    except ValueError:
        return None


@lru_cache(maxsize=8)
def _graph_catalogue(
    tokens: tuple[str, ...],
    room_labels: tuple[str, ...],
    max_rooms: int,
    max_walls: int,
    max_openings: int,
) -> dict[str, torch.Tensor]:
    size = len(tokens) + 1
    adjacency = torch.zeros(size, max_rooms, max_rooms)
    room_label = torch.zeros(size, max_rooms, len(room_labels))
    wall_exterior = torch.zeros(size, max_walls)
    wall_interior = torch.zeros(size, max_walls)
    opening_host = torch.zeros(size, max_openings, max_walls)
    label_lookup = {name: index for index, name in enumerate(room_labels)}
    for token_id, value in enumerate(tokens):
        token = FloorplanToken.parse(value)
        if token.kind == "ADJACENCY" and len(token.args) >= 2:
            left = _entity_index(token.args[0], "room_")
            right = _entity_index(token.args[1], "room_")
            if (
                left is not None
                and right is not None
                and 0 <= left < max_rooms
                and 0 <= right < max_rooms
            ):
                adjacency[token_id, left, right] = 1.0
                adjacency[token_id, right, left] = 1.0
        elif token.kind == "ROOM_BEGIN" and len(token.args) >= 2:
            room = _entity_index(token.args[0], "room_")
            label = label_lookup.get(token.args[1])
            if room is not None and label is not None and 0 <= room < max_rooms:
                room_label[token_id, room, label] = 1.0
        elif token.kind == "WALL" and token.args:
            wall = _entity_index(token.args[0], "wall_")
            wall_type = token.args[-1]
            if wall is not None and 0 <= wall < max_walls:
                if wall_type == "exterior":
                    wall_exterior[token_id, wall] = 1.0
                else:
                    wall_interior[token_id, wall] = 1.0
        elif token.kind == "OPENING" and len(token.args) >= 3:
            opening = _entity_index(token.args[0], "opening_")
            wall = _entity_index(token.args[2], "wall_")
            if (
                opening is not None
                and wall is not None
                and 0 <= opening < max_openings
                and 0 <= wall < max_walls
            ):
                opening_host[token_id, opening, wall] = 1.0
    return {
        "adjacency": adjacency,
        "room_label": room_label,
        "wall_exterior": wall_exterior,
        "wall_interior": wall_interior,
        "opening_host": opening_host,
    }


class TokenSoftGraphCatalogue(nn.Module):
    def __init__(
        self,
        vocabulary: FloorplanTokenVocabulary,
        room_labels: Sequence[str],
        *,
        max_rooms: int,
        max_walls: int,
        max_openings: int,
    ) -> None:
        super().__init__()
        catalogue = _graph_catalogue(
            tuple(vocabulary.tokens),
            tuple(str(value) for value in room_labels),
            int(max_rooms),
            int(max_walls),
            int(max_openings),
        )
        for name, value in catalogue.items():
            self.register_buffer(name, value, persistent=False)

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "adjacency": self.adjacency,
            "room_label": self.room_label,
            "wall_exterior": self.wall_exterior,
            "wall_interior": self.wall_interior,
            "opening_host": self.opening_host,
        }


def configure_token_graph_catalogue(model, vocabulary, taxonomy) -> None:
    if model.token_head is None:
        return
    model.token_graph_catalogue = TokenSoftGraphCatalogue(
        vocabulary,
        tuple(
            taxonomy.class_names[class_id]
            for class_id in taxonomy.room_ids
        ),
        max_rooms=model.config.identity_queries,
        max_walls=model.config.max_walls,
        max_openings=model.config.max_openings,
    ).to(next(model.parameters()).device)


def token_soft_graph_consistency_loss(
    token_logits: torch.Tensor,
    vocabulary: FloorplanTokenVocabulary,
    predictions: dict[str, torch.Tensor],
    *,
    token_valid: torch.Tensor,
    room_valid: torch.Tensor,
    wall_valid: torch.Tensor,
    opening_valid: torch.Tensor,
    room_labels: Sequence[str],
    catalogue: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare vector graph probabilities with graph probabilities implied by token logits."""

    adjacency_logits = predictions["room_adjacency_logits"]
    max_rooms = adjacency_logits.shape[-1]
    max_walls = predictions["wall_exterior_logits"].shape[-1]
    max_openings = predictions["opening_host_logits"].shape[-2]
    catalogue = catalogue or _graph_catalogue(
        tuple(vocabulary.tokens),
        tuple(str(value) for value in room_labels),
        int(max_rooms),
        int(max_walls),
        int(max_openings),
    )
    device = token_logits.device
    if token_valid.shape != token_logits.shape[:2]:
        raise ValueError("token_valid must match token logits")
    probabilities = token_logits.softmax(dim=-1) * token_valid[..., None]
    adjacency_positions = torch.einsum(
        "blv,vij->blij",
        probabilities,
        catalogue["adjacency"].to(device),
    )
    adjacency_targets = 1.0 - (1.0 - adjacency_positions).prod(dim=1)
    pair_mask = room_valid[:, :, None] & room_valid[:, None, :]
    adjacency_loss = F.mse_loss(
        adjacency_logits.sigmoid()[pair_mask],
        adjacency_targets[pair_mask],
    )
    room_targets = torch.einsum(
        "blv,vqc->bqc",
        probabilities,
        catalogue["room_label"].to(device),
    )
    room_mass = room_targets.sum(dim=-1)
    room_targets = room_targets / room_mass[..., None].clamp_min(1.0e-6)
    room_mask = room_valid & (room_mass > 1.0e-6)
    room_label_loss = F.mse_loss(
        predictions["room_class"].softmax(dim=-1)[room_mask],
        room_targets[room_mask],
    )
    exterior_mass = torch.einsum(
        "blv,vw->bw",
        probabilities,
        catalogue["wall_exterior"].to(device),
    )
    interior_mass = torch.einsum(
        "blv,vw->bw",
        probabilities,
        catalogue["wall_interior"].to(device),
    )
    wall_mass = exterior_mass + interior_mass
    wall_targets = exterior_mass / wall_mass.clamp_min(1.0e-6)
    wall_mask = wall_valid & (wall_mass > 1.0e-6)
    wall_type_loss = F.mse_loss(
        predictions["wall_exterior_logits"].sigmoid()[wall_mask],
        wall_targets[wall_mask],
    )
    host_targets = torch.einsum(
        "blv,vow->bow",
        probabilities,
        catalogue["opening_host"].to(device),
    )
    host_mass = host_targets.sum(dim=-1)
    host_targets = host_targets / host_mass[..., None].clamp_min(1.0e-6)
    host_mask = opening_valid & (host_mass > 1.0e-6)
    opening_host_loss = F.mse_loss(
        predictions["opening_host_logits"].softmax(dim=-1)[host_mask],
        host_targets[host_mask],
    )
    total = (
        adjacency_loss
        + room_label_loss
        + wall_type_loss
        + opening_host_loss
    )
    return total, {
        "token_implied_adjacency_consistency": adjacency_loss.detach(),
        "token_implied_room_label_consistency": room_label_loss.detach(),
        "token_implied_wall_type_consistency": wall_type_loss.detach(),
        "token_implied_opening_host_consistency": opening_host_loss.detach(),
    }


__all__ = [
    "TOKEN_SOFT_GRAPH_VERSION",
    "TokenSoftGraphCatalogue",
    "configure_token_graph_catalogue",
    "token_soft_graph_consistency_loss",
]

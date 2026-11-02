"""Contour-conditioned room-program presence and partition bias."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy


class RoomProgramHead(nn.Module):
    """Predict the room-class set and expose per-class local partition biases."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_dim: int = 96,
        tokens: int = 8,
        layers: int = 1,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.num_room_classes = int(num_room_classes)
        self.tokens = int(tokens)
        self.project = nn.Conv2d(in_channels, hidden_dim, 1)
        self.position = nn.Parameter(torch.zeros(1, tokens * tokens, hidden_dim))
        self.class_queries = nn.Parameter(torch.randn(1, num_room_classes, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    hidden_dim,
                    heads,
                    dim_feedforward=hidden_dim * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(max(1, int(layers)))
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.presence = nn.Linear(hidden_dim, 1)
        self.bias = nn.Linear(hidden_dim, 1)
        for module in (self.presence, self.bias):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.project(features)
        hidden = F.adaptive_avg_pool2d(hidden, (self.tokens, self.tokens))
        memory = hidden.flatten(2).transpose(1, 2) + self.position.to(hidden.dtype)
        query = self.class_queries.to(hidden.dtype).expand(features.shape[0], -1, -1)
        for layer in self.layers:
            query = layer(query, memory)
        query = self.norm(query)
        return {
            "presence_logits": self.presence(query).squeeze(-1),
            "class_bias": self.bias(query).squeeze(-1),
            "class_tokens": query,
        }


def room_presence_target(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    return torch.stack(
        [
            torch.bincount(
                item.flatten(),
                minlength=taxonomy.num_classes,
            )[ids].gt(0).float()
            for item in labels
        ],
        dim=0,
    )


def room_count_target(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    return torch.stack(
        [
            torch.bincount(
                item.flatten(),
                minlength=taxonomy.num_classes,
            )[ids]
            for item in labels
        ],
        dim=0,
    ).long()


def gate_room_labels_to_program(
    labels: torch.Tensor,
    program: torch.Tensor,
    room_class_ids: tuple[int, ...],
) -> torch.Tensor:
    """Remove room classes that an explicit program declares absent."""

    present = program.to(labels.device) > 0.5
    if present.ndim == 1:
        present = present[None]
    if present.shape != (labels.shape[0], len(room_class_ids)):
        raise ValueError("program must match batch and room-class dimensions")
    lookup_size = max(labels.max().item(), max(room_class_ids)) + 1
    lookup = torch.full(
        (lookup_size,),
        -1,
        dtype=torch.long,
        device=labels.device,
    )
    lookup[torch.tensor(room_class_ids, device=labels.device)] = torch.arange(
        len(room_class_ids),
        device=labels.device,
    )
    class_index = lookup[labels]
    is_room = class_index >= 0
    present_per_pixel = present[:, :, None, None].expand(
        -1,
        -1,
        labels.shape[-2],
        labels.shape[-1],
    )
    allowed = torch.gather(
        present_per_pixel,
        1,
        class_index.clamp_min(0)[:, None],
    )[:, 0]
    allowed = is_room & allowed
    return torch.where(allowed | ~is_room, labels, torch.zeros_like(labels))


@torch.no_grad()
def cardinality_gated_program(
    presence_logits: torch.Tensor,
    *,
    minimum_rooms: int = 1,
) -> torch.Tensor:
    """Choose room presence without a fixed threshold or target room count."""

    probability = presence_logits.float().sigmoid()
    count = probability.sum(dim=1).round().long().clamp(
        min(int(minimum_rooms), probability.shape[1]),
        probability.shape[1],
    )
    indices = probability.topk(int(count.max()), dim=1).indices
    selected = torch.zeros_like(probability, dtype=torch.bool)
    positions = torch.arange(count.max(), device=probability.device)
    selected.scatter_(
        1,
        indices,
        positions[None, :] < count[:, None],
    )
    return selected


def room_program_loss(
    predictions: dict[str, torch.Tensor],
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = room_presence_target(labels, taxonomy)
    logits = predictions["presence_logits"]
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=class_weights,
    )
    return loss, {
        "room_program_loss": loss.detach(),
        "room_program_positive_fraction": target.mean().detach(),
    }

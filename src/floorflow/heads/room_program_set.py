"""Cardinality-aware room-program prediction as a query set."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


class RoomProgramSetHead(nn.Module):
    """Predict room classes as an unordered set, including repeated room types."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_dim: int = 96,
        tokens: int = 8,
        layers: int = 2,
        heads: int = 4,
        num_queries: int = 32,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.num_room_classes = int(num_room_classes)
        self.tokens = int(tokens)
        self.project = nn.Conv2d(in_channels, hidden_dim, 1)
        self.position = nn.Parameter(torch.zeros(1, tokens * tokens, hidden_dim))
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
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
        self.presence_queries = nn.Parameter(
            torch.randn(1, self.num_room_classes, hidden_dim) * 0.02
        )
        self.presence_layers = nn.ModuleList(
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
        self.presence_norm = nn.LayerNorm(hidden_dim)
        self.class_head = nn.Linear(hidden_dim, self.num_room_classes + 1)
        self.presence_class_head = nn.Linear(hidden_dim, 1)
        self.cardinality_head = nn.Linear(hidden_dim * 2, self.num_room_classes + 1)
        self.query_cardinality_residual = nn.Linear(
            hidden_dim * 2,
            self.num_room_classes + 1,
        )
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)
        nn.init.xavier_uniform_(self.presence_class_head.weight)
        nn.init.zeros_(self.presence_class_head.bias)
        nn.init.zeros_(self.query_cardinality_residual.weight)
        nn.init.zeros_(self.query_cardinality_residual.bias)
        self.register_buffer(
            "class_presence_enabled",
            torch.tensor(True),
            persistent=True,
        )
        self.register_buffer(
            "program_prior",
            torch.ones(self.num_room_classes),
            persistent=True,
        )
        self.register_buffer(
            "program_ranking_strength",
            torch.tensor(-1.0),
            persistent=True,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.project(features)
        hidden = F.adaptive_avg_pool2d(hidden, (self.tokens, self.tokens))
        memory = hidden.flatten(2).transpose(1, 2) + self.position.to(hidden.dtype)
        query = self.queries.to(hidden.dtype).expand(features.shape[0], -1, -1)
        for layer in self.layers:
            query = layer(query, memory)
        query = self.norm(query)
        presence_query = self.presence_queries.to(hidden.dtype).expand(
            features.shape[0],
            -1,
            -1,
        )
        for layer in self.presence_layers:
            presence_query = layer(presence_query, memory)
        presence_query = self.presence_norm(presence_query)
        memory_summary = torch.cat(
            [memory.mean(dim=1), memory.amax(dim=1)],
            dim=-1,
        )
        query_summary = torch.cat(
            [query.mean(dim=1), query.amax(dim=1)],
            dim=-1,
        )
        class_logits = self.class_head(query)
        presence_logits = (
            self.presence_class_head(presence_query).squeeze(-1)
            if bool(self.class_presence_enabled)
            else program_logits_from_queries(class_logits)
        )
        rank_logits = (
            presence_logits
            + self.program_ranking_strength.to(presence_logits.dtype)
            * self.program_prior.to(presence_logits.dtype).log()[None]
        )
        cardinality_logits = (
            self.cardinality_head(memory_summary)
            + self.query_cardinality_residual(query_summary)
        )
        return {
            "class_logits": class_logits,
            "cardinality_logits": cardinality_logits,
            "presence_logits": presence_logits,
            "presence_rank_logits": rank_logits,
            "class_tokens": query,
            "presence_tokens": presence_query,
        }


def program_logits_from_queries(class_logits: torch.Tensor) -> torch.Tensor:
    """Aggregate instance queries into a duplicate-insensitive class presence logit."""

    foreground = class_logits[..., :-1]
    return torch.logsumexp(foreground, dim=1)


def load_room_program_set_state(
    head: RoomProgramSetHead,
    state: dict[str, torch.Tensor],
) -> None:
    """Load old set-head checkpoints with the zero residual extension."""

    missing, unexpected = head.load_state_dict(state, strict=False)
    presence_prefixes = (
        "presence_queries",
        "presence_layers.",
        "presence_norm.",
        "presence_class_head.",
        "class_presence_enabled",
        "program_prior",
        "program_ranking_strength",
    )
    allowed = {
        name
        for name in missing
        if name.startswith("query_cardinality_residual.")
        or name.startswith(presence_prefixes)
    }
    if len(allowed) != len(missing) or unexpected:
        raise RuntimeError(
            f"program set state mismatch: missing={missing}, unexpected={unexpected}"
        )
    if any(name.startswith(presence_prefixes) for name in missing):
        head.class_presence_enabled.fill_(False)


def match_room_program(
    class_logits: torch.Tensor,
    target_classes: torch.Tensor,
    target_valid: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    rows = []
    columns = []
    probability = class_logits.softmax(dim=-1)
    for index in range(target_classes.shape[0]):
        count = int(target_valid[index].sum())
        if count == 0:
            rows.append(torch.empty(0, dtype=torch.long, device=class_logits.device))
            columns.append(torch.empty(0, dtype=torch.long, device=class_logits.device))
            continue
        cost = -probability[index][:, target_classes[index, :count]]
        matched_rows, matched_columns = linear_sum_assignment(
            cost.detach().float().cpu().numpy()
        )
        rows.append(torch.as_tensor(matched_rows, device=class_logits.device))
        columns.append(torch.as_tensor(matched_columns, device=class_logits.device))
    return rows, columns


def room_program_cooccurrence_loss(
    presence_logits: torch.Tensor,
    target_presence: torch.Tensor,
    *,
    positive_weight: float = 2.0,
) -> torch.Tensor:
    """Match the predicted joint presence of every room-class pair."""

    classes = presence_logits.shape[1]
    rows, columns = torch.triu_indices(
        classes,
        classes,
        offset=1,
        device=presence_logits.device,
    )
    probability = presence_logits.sigmoid()
    pair_probability = probability[:, rows] * probability[:, columns]
    pair_target = target_presence[:, rows] * target_presence[:, columns]
    weight = 1.0 + float(positive_weight) * pair_target
    return F.binary_cross_entropy(
        pair_probability.clamp(1.0e-6, 1.0 - 1.0e-6),
        pair_target,
        weight=weight,
    )


def room_program_set_loss(
    predictions: dict[str, torch.Tensor],
    target_classes: torch.Tensor,
    target_valid: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    no_object_weight: float = 0.1,
    cardinality_weight: float = 0.5,
    presence_weight: float = 1.0,
    cooccurrence_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Hungarian class loss plus explicit room-class cardinality supervision."""

    class_logits = predictions["class_logits"]
    rows, columns = match_room_program(class_logits, target_classes, target_valid)
    no_object_class = class_logits.shape[-1] - 1
    assigned = torch.full(
        class_logits.shape[:2],
        no_object_class,
        dtype=torch.long,
        device=class_logits.device,
    )
    for index, (matched_rows, matched_columns) in enumerate(zip(rows, columns)):
        if matched_rows.numel():
            assigned[index, matched_rows] = target_classes[index, matched_columns]
    weights = (
        torch.cat(
            [
                class_weights,
                torch.ones(1, device=class_weights.device, dtype=class_weights.dtype),
            ]
        )
        if class_weights is not None
        else torch.ones(class_logits.shape[-1], device=class_logits.device)
    )
    if no_object_weight != 1.0:
        weights = weights.clone()
        weights[-1] = float(no_object_weight)
    set_loss = F.cross_entropy(
        class_logits.transpose(1, 2),
        assigned,
        weight=weights,
    )
    target_presence = torch.zeros(
        target_classes.shape[0],
        class_logits.shape[-1] - 1,
        device=class_logits.device,
    )
    for index in range(target_classes.shape[0]):
        count = int(target_valid[index].sum())
        if count:
            target_presence[index, target_classes[index, :count]] = 1.0
    presence_loss = F.binary_cross_entropy_with_logits(
        predictions["presence_logits"],
        target_presence,
        pos_weight=class_weights,
    )
    cardinality_target = target_presence.sum(dim=1).long()
    cardinality_loss = F.cross_entropy(
        predictions["cardinality_logits"],
        cardinality_target,
    )
    cooccurrence_loss = room_program_cooccurrence_loss(
        predictions["presence_logits"],
        target_presence,
    )
    total = (
        set_loss
        + float(presence_weight) * presence_loss
        + float(cardinality_weight) * cardinality_loss
        + float(cooccurrence_weight) * cooccurrence_loss
    )
    return total, {
        "room_program_set_loss": set_loss.detach(),
        "room_program_presence_loss": presence_loss.detach(),
        "room_program_cardinality_loss": cardinality_loss.detach(),
        "room_program_cooccurrence_loss": cooccurrence_loss.detach(),
        "room_program_total_loss": total.detach(),
    }


@torch.no_grad()
def decode_room_program(
    predictions: dict[str, torch.Tensor],
    *,
    cardinality_method: str = "head",
    oracle_count: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select a class set using predicted cardinality instead of class thresholds."""

    ranking = predictions.get(
        "presence_rank_logits",
        predictions["presence_logits"],
    )
    presence = predictions["presence_logits"]
    class_logits = predictions["class_logits"]
    if oracle_count is not None:
        count = oracle_count
    elif cardinality_method == "head":
        count = predictions["cardinality_logits"].argmax(dim=-1)
    elif cardinality_method == "expected":
        count = presence.sigmoid().sum(dim=-1).round().long()
    elif cardinality_method == "query_count":
        no_object = class_logits.shape[-1] - 1
        count = (class_logits.argmax(dim=-1) != no_object).sum(dim=-1)
    elif cardinality_method == "query_mass":
        class_logits = class_logits.float()
        count = (
            1.0 - class_logits[..., -1].softmax(dim=-1)
        ).sum(dim=-1).round().long()
    else:
        raise ValueError(
            "cardinality_method must be head, expected, query_count, or query_mass"
        )
    count = count.clamp_min(1)
    count = count.clamp_max(ranking.shape[1])
    indices = ranking.topk(int(count.max()), dim=1).indices
    selected = torch.zeros_like(ranking, dtype=torch.bool)
    positions = torch.arange(count.max(), device=ranking.device)
    selected.scatter_(1, indices, positions[None, :] < count[:, None])
    return selected


def program_set_metrics(
    predictions: dict[str, torch.Tensor],
    target_presence: torch.Tensor,
    *,
    cardinality_method: str = "head",
) -> dict[str, float]:
    oracle_count = (
        target_presence.sum(dim=1).long()
        if cardinality_method == "oracle"
        else None
    )
    predicted = decode_room_program(
        predictions,
        cardinality_method="head" if oracle_count is not None else cardinality_method,
        oracle_count=oracle_count,
    )
    expected = target_presence.bool()
    tp = (predicted & expected).sum().float()
    precision = tp / predicted.sum().clamp_min(1.0)
    recall = tp / expected.sum().clamp_min(1.0)
    if oracle_count is not None:
        cardinality = oracle_count
    elif cardinality_method == "head":
        cardinality = predictions["cardinality_logits"].argmax(dim=-1)
    elif cardinality_method == "expected":
        cardinality = predictions["presence_logits"].sigmoid().sum(dim=-1).round().long()
    elif cardinality_method == "query_count":
        no_object = predictions["class_logits"].shape[-1] - 1
        cardinality = (
            predictions["class_logits"].argmax(dim=-1) != no_object
        ).sum(dim=-1)
    else:
        class_probability = predictions["class_logits"].softmax(dim=-1)
        cardinality = (
            1.0 - class_probability[..., -1]
        ).sum(dim=-1).round().long()
    return {
        "presence_precision": float(precision.cpu()),
        "presence_recall": float(recall.cpu()),
        "presence_f1": float(
            (
                2.0
                * precision
                * recall
                / (precision + recall).clamp_min(1.0e-6)
            ).cpu()
        ),
        "exact_set_accuracy": float((predicted == expected).all(dim=1).float().mean().cpu()),
        "cardinality_accuracy": float(
            (cardinality == expected.sum(dim=1)).float().mean().cpu()
        ),
    }


__all__ = [
    "RoomProgramSetHead",
    "decode_room_program",
    "load_room_program_set_state",
    "match_room_program",
    "program_logits_from_queries",
    "program_set_metrics",
    "room_program_cooccurrence_loss",
    "room_program_set_loss",
]

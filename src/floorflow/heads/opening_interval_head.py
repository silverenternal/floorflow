"""Learned wall-aligned opening intervals and door/window type logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WallIntervalHead(nn.Module):
    """Cross-attend openings to wall tokens and regress 1D wall intervals."""

    def __init__(
        self,
        hidden_dim: int,
        num_types: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim % heads:
            heads = 1
        self.wall_norm = nn.LayerNorm(hidden_dim)
        self.opening_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            batch_first=True,
        )
        self.interval = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.type_head = nn.Linear(hidden_dim, int(num_types))
        nn.init.zeros_(self.interval[-1].weight)
        nn.init.zeros_(self.interval[-1].bias)
        nn.init.zeros_(self.type_head.weight)
        nn.init.zeros_(self.type_head.bias)

    def forward(
        self,
        wall_tokens: torch.Tensor,
        opening_tokens: torch.Tensor,
        wall_endpoints: torch.Tensor,
        *,
        wall_mask: torch.Tensor | None = None,
        opening_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if wall_tokens.ndim != 3 or opening_tokens.ndim != 3:
            raise ValueError("wall and opening tokens must be [B,N,D]")
        if wall_tokens.shape[0] != opening_tokens.shape[0]:
            raise ValueError("wall and opening batches must match")
        if wall_endpoints.shape[:2] != wall_tokens.shape[:2] or wall_endpoints.shape[-1] != 4:
            raise ValueError("wall_endpoints must be [B,W,4]")
        empty = opening_tokens.shape[1] == 0
        if opening_mask is not None and opening_tokens.shape[1]:
            empty = empty or bool((opening_mask.sum(dim=1) == 0).any())
        if empty:
            attended = torch.zeros_like(wall_tokens)
        else:
            key_padding = None if opening_mask is None else ~opening_mask.bool()
            attended = self.cross_attention(
                self.wall_norm(wall_tokens),
                self.opening_norm(opening_tokens),
                self.opening_norm(opening_tokens),
                key_padding_mask=key_padding,
                need_weights=False,
            )[0]
        hidden = wall_tokens + attended
        raw = self.interval(hidden)
        presence_logits = raw[..., 0]
        start = raw[..., 1].sigmoid()
        length = raw[..., 2].sigmoid() * (1.0 - start)
        center = (start + 0.5 * length).clamp(0.0, 1.0)
        start_xy = wall_endpoints[..., :2]
        end_xy = wall_endpoints[..., 2:]
        center_xy = start_xy + center[..., None] * (end_xy - start_xy)
        interval_start_xy = start_xy + start[..., None] * (end_xy - start_xy)
        interval_end_xy = start_xy + (start + length)[..., None] * (end_xy - start_xy)
        if wall_mask is not None:
            valid = wall_mask.to(hidden.dtype)
            presence_logits = presence_logits * valid
        return {
            "presence_logits": presence_logits,
            "start_fraction": start,
            "length_fraction": length,
            "type_logits": self.type_head(hidden),
            "center_xy": center_xy,
            "start_xy": interval_start_xy,
            "end_xy": interval_end_xy,
        }


def interval_targets_from_plan(
    plan,
    *,
    num_walls: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build per-wall presence/start/length/type targets from a readout plan."""

    import math

    count = int(num_walls if num_walls is not None else len(plan.walls))
    presence = torch.zeros(count)
    start = torch.zeros(count)
    length = torch.zeros(count)
    type_ids = torch.zeros(count, dtype=torch.long)
    for opening in plan.openings:
        index = int(opening.wall_index)
        if not 0 <= index < count:
            continue
        wall = plan.walls[index]
        wall_length = max(
            1.0e-6,
            math.dist(wall.start, wall.end),
        )
        projection = _project_fraction(opening.center, wall)
        width = min(float(opening.width), wall_length)
        presence[index] = 1.0
        start[index] = max(
            0.0,
            min(1.0, projection - width / (2.0 * wall_length)),
        )
        length[index] = min(
            1.0 - start[index],
            width / wall_length,
        )
        type_ids[index] = 0 if opening.kind == "door" else 1
    return {
        "presence": presence,
        "start": start,
        "length": length,
        "type": type_ids,
    }


def _project_fraction(
    point: tuple[float, float],
    wall,
) -> float:
    import numpy as np

    start = np.asarray(wall.start, dtype=np.float64)
    end = np.asarray(wall.end, dtype=np.float64)
    delta = end - start
    length_sq = float(delta @ delta)
    if length_sq <= 1.0e-9:
        return 0.0
    return float(
        np.clip((np.asarray(point) - start) @ delta / length_sq, 0.0, 1.0)
    )


def wall_interval_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    presence_weight: float = 1.0,
    geometry_weight: float = 1.0,
    type_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Presence BCE + interval start/length L1 + door/window CE."""

    presence = targets["presence"].to(predictions["presence_logits"].device)
    start = targets["start"].to(predictions["start_fraction"].device)
    length = targets["length"].to(predictions["length_fraction"].device)
    type_ids = targets["type"].to(predictions["type_logits"].device)
    if presence.ndim == 1 and predictions["presence_logits"].ndim == 2:
        presence = presence[None].expand_as(predictions["presence_logits"])
        start = start[None].expand_as(predictions["start_fraction"])
        length = length[None].expand_as(predictions["length_fraction"])
        type_ids = type_ids[None].expand(
            predictions["type_logits"].shape[:-1]
        )
    presence_loss = F.binary_cross_entropy_with_logits(
        predictions["presence_logits"],
        presence,
    )
    positive = presence > 0.5
    if positive.any():
        start_loss = F.l1_loss(
            predictions["start_fraction"][positive],
            start[positive],
        )
        length_loss = F.l1_loss(
            predictions["length_fraction"][positive],
            length[positive],
        )
        type_loss = F.cross_entropy(
            predictions["type_logits"][positive],
            type_ids[positive],
        )
    else:
        zero = presence_loss * 0.0
        start_loss = zero
        length_loss = zero
        type_loss = zero
    total = (
        float(presence_weight) * presence_loss
        + float(geometry_weight) * (start_loss + length_loss)
        + float(type_weight) * type_loss
    )
    return total, {
        "opening_interval_presence_loss": presence_loss.detach(),
        "opening_interval_start_loss": start_loss.detach(),
        "opening_interval_length_loss": length_loss.detach(),
        "opening_interval_type_loss": type_loss.detach(),
    }


__all__ = [
    "WallIntervalHead",
    "interval_targets_from_plan",
    "wall_interval_loss",
]

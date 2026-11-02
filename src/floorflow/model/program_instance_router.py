"""Program-schedule routing into room-instance query classes."""

from __future__ import annotations

import math

import torch
from torch import nn


class ProgramInstanceRouter(nn.Module):
    """Condition query-to-room-class assignment on the requested program."""

    def __init__(
        self,
        hidden_dim: int,
        num_room_classes: int,
    ) -> None:
        super().__init__()
        self.num_room_classes = int(num_room_classes)
        self.query_projection = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.program_projection = nn.Linear(
            self.num_room_classes,
            int(hidden_dim),
        )
        self.class_embedding = nn.Parameter(
            torch.randn(self.num_room_classes, hidden_dim) * 0.02
        )
        self.class_bias = nn.Parameter(torch.zeros(self.num_room_classes))
        self.background = nn.Linear(hidden_dim * 2, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.background.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.background.bias)

    def forward(
        self,
        query_features: torch.Tensor,
        program: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if query_features.ndim != 3:
            raise ValueError("query features must be [B,Q,D]")
        if program.shape != (
            query_features.shape[0],
            self.num_room_classes,
        ):
            raise ValueError("program width does not match room classes")
        request = program.float().clamp(0.0, 1.0)
        class_tokens = (
            self.class_embedding[None]
            + self.program_projection(request)[:, None]
        )
        query = self.query_projection(query_features)
        foreground = torch.einsum(
            "bqd,bcd->bqc",
            query,
            class_tokens,
        ) / math.sqrt(float(query.shape[-1]))
        foreground = foreground + self.class_bias[None, None]
        request_context = self.program_projection(request)[:, None].expand(
            -1,
            query_features.shape[1],
            -1,
        )
        background = self.background(
            torch.cat([query_features, request_context], dim=-1)
        )
        logits = torch.cat([background, foreground], dim=-1)
        room_class_logits = torch.cat([foreground, background], dim=-1)
        scheduled = foreground.masked_fill(
            request[:, None] <= 0.0,
            -1.0e4,
        )
        scheduled_room_class = torch.cat(
            [scheduled, logits[..., :1]],
            dim=-1,
        )
        return {
            "assignment_logits": logits,
            "room_class_assignment_logits": room_class_logits,
            "foreground_assignment_logits": foreground,
            "scheduled_assignment_logits": scheduled,
            "scheduled_room_class_logits": scheduled_room_class,
            "assignment_scale": torch.tanh(self.scale).expand(
                query_features.shape[0]
            ),
        }


__all__ = ["ProgramInstanceRouter"]

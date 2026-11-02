"""Room-schedule-conditioned semantic residuals for furniture queries."""

from __future__ import annotations

import torch
from torch import nn


class RoomConditionedFurnitureSemantics(nn.Module):
    """Refine furniture type logits using room-type evidence and schedule."""

    def __init__(
        self,
        query_dim: int,
        num_room_classes: int,
        num_furniture_classes: int,
        *,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(int(query_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(num_furniture_classes)),
        )
        self.room_projection = nn.Sequential(
            nn.Linear(int(num_room_classes), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(num_furniture_classes)),
        )
        self.room_furniture_prior = nn.Parameter(
            torch.randn(
                int(num_room_classes),
                int(num_furniture_classes),
            )
            * 1.0e-3
        )
        self.gate = nn.Sequential(
            nn.Linear(
                int(query_dim) + int(num_room_classes),
                hidden,
            ),
            nn.SiLU(),
            nn.Linear(hidden, int(num_furniture_classes)),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))
        for module in (
            self.query_projection[-1],
            self.room_projection[-1],
            self.gate[-1],
        ):
            nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        query_features: torch.Tensor,
        room_presence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if query_features.ndim != 3:
            raise ValueError("query features must be [B,Q,D]")
        if room_presence.shape != (
            query_features.shape[0],
            self.room_furniture_prior.shape[0],
        ):
            raise ValueError("room presence does not match query batch")
        room_context = room_presence @ self.room_furniture_prior
        residual = (
            self.query_projection(query_features)
            + self.room_projection(room_presence)[:, None]
            + room_context[:, None]
        )
        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [
                        query_features,
                        room_presence[:, None].expand(
                            -1,
                            query_features.shape[1],
                            -1,
                        ),
                    ],
                    dim=-1,
                )
            )
        )
        residual = torch.tanh(self.scale) * gate * residual
        return {
            "class_residual": residual,
            "semantic_gate": gate,
            "room_context": room_context,
        }


__all__ = ["RoomConditionedFurnitureSemantics"]

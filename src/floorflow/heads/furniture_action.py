"""Discrete local actions for furniture tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

ACTIONS = ("keep", "translate", "rotate", "swap", "drop")


class FurnitureActionHead(nn.Module):
    def __init__(self, token_dim: int, room_dim: int | None = None) -> None:
        super().__init__()
        room_dim = int(room_dim or token_dim)
        self.net = nn.Sequential(
            nn.Linear(token_dim + room_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, len(ACTIONS) + 3),
        )

    def forward(self, furniture: torch.Tensor, room: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.net(torch.cat([furniture, room], dim=-1))
        return output[..., : len(ACTIONS)], output[..., len(ACTIONS) :]


@torch.no_grad()
def apply_furniture_action(
    center: torch.Tensor,
    action: torch.Tensor,
    transform: torch.Tensor,
    *,
    max_translate: float = 0.15,
) -> torch.Tensor:
    moved = center.clone()
    name = ACTIONS[int(action)]
    if name == "translate":
        moved = moved + transform[..., :2].tanh() * max_translate
    elif name == "rotate":
        angle = transform[..., 2].tanh() * 0.5
        cosine, sine = torch.cos(angle), torch.sin(angle)
        x, y = moved.unbind(-1)
        moved = torch.stack([cosine * x - sine * y, sine * x + cosine * y], dim=-1)
    return moved

"""Spatial-to-slot gated velocity head."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialVelocityHead(nn.Module):
    def __init__(self, hidden_dim: int, stream_names: tuple[str, ...], output_dim: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.keys = nn.ModuleDict()
        self.values = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))
        self.heads = int(heads)
        for name in stream_names:
            self.keys[name] = nn.Linear(hidden_dim, hidden_dim)
            self.values[name] = nn.Linear(hidden_dim, hidden_dim)
            self.gates[name] = nn.Linear(hidden_dim * 2, 1)

    def forward(self, spatial: torch.Tensor, slots: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query = self.query(spatial)
        pieces = []
        diagnostics = {}
        for name, slot in slots.items():
            if name not in self.keys:
                raise KeyError(f"unregistered spatial velocity stream: {name}")
            scores = torch.einsum("bnd,bmd->bnm", query, self.keys[name](slot)) / query.shape[-1] ** 0.5
            attention = torch.softmax(scores, dim=-1)
            context = torch.einsum("bnm,bmd->bnd", attention, self.values[name](slot))
            gate = torch.sigmoid(self.gates[name](torch.cat([spatial, context], dim=-1)))
            pieces.append(gate * context)
            diagnostics[f"gate_{name}"] = gate.mean()
        velocity = self.output(torch.stack(pieces, dim=0).mean(dim=0))
        return velocity, diagnostics

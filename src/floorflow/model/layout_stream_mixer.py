"""Normalized gate mixer for floorplan semantic streams."""

from __future__ import annotations

import torch
import torch.nn as nn

from floorflow.representation.gate_diagnostics import stream_gate_diagnostics

class LayoutStreamMixer(nn.Module):
    def __init__(self, hidden_dim: int, stream_names: tuple[str, ...]) -> None:
        super().__init__()
        self.stream_names = stream_names
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.keys = nn.ModuleDict({name: nn.Linear(hidden_dim, hidden_dim) for name in stream_names})
        self.values = nn.ModuleDict({name: nn.Linear(hidden_dim, hidden_dim) for name in stream_names})
        self.gates = nn.ModuleDict({name: nn.Linear(hidden_dim * 2, 1) for name in stream_names})
        self.output = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_tokens: torch.Tensor, streams: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(query_tokens)
        gate_values = []
        contexts = []
        for name in self.stream_names:
            if name not in streams:
                continue
            tokens = streams[name]
            scores = torch.einsum("bnd,bmd->bnm", query, self.keys[name](tokens)) / query.shape[-1] ** 0.5
            attention = torch.softmax(scores, dim=-1)
            context = torch.einsum("bnm,bmd->bnd", attention, self.values[name](tokens))
            gate_values.append(torch.sigmoid(self.gates[name](torch.cat([query_tokens, context], dim=-1))))
            contexts.append(gate_values[-1] * context)
        gates = torch.cat(gate_values, dim=-1)
        normalized = torch.softmax(gates, dim=-1)
        mixed = sum(normalized[..., index : index + 1] * context for index, context in enumerate(contexts))
        output = self.norm(query_tokens + self.output(torch.cat([query_tokens, mixed], dim=-1)))
        return output, normalized

"""Explicit gated attention between room, wall, opening, and furniture streams."""

from __future__ import annotations

import torch
import torch.nn as nn

from floorflow.representation.gate_diagnostics import stream_gate_diagnostics
from floorflow.model.mixture_topk_attention import MixtureTopKAttention
from floorflow.model.official_mita_route import OfficialMiTARouteAttention


class SemanticFamilyStreamMixer(nn.Module):
    """Split one hidden map into semantic families, then exchange context."""

    def __init__(
        self,
        hidden_channels: int,
        stream_names: tuple[str, ...] = ("room", "wall", "opening", "furniture"),
        heads: int = 4,
        dropout: float = 0.0,
        attention_type: str = "mha",
        attention_impl: str = "legacy",
        router_count: int = 16,
        top_k: int = 8,
    ) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if not stream_names:
            raise ValueError("at least one semantic stream is required")
        heads = int(heads)
        if hidden_channels % heads:
            heads = 1
        self.stream_names = tuple(stream_names)
        self.hidden_channels = hidden_channels
        self.attention_type = str(attention_type)
        if self.attention_type not in {"mha", "mita"}:
            raise ValueError("attention_type must be 'mha' or 'mita'")
        if attention_impl not in {"legacy", "official"}:
            raise ValueError("attention_impl must be 'legacy' or 'official'")
        self.attention_impl = str(attention_impl) if attention_type == "mita" else "mha"
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 1),
                    nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
                    nn.SiLU(),
                )
                for name in self.stream_names
            }
        )
        self.attention = nn.ModuleDict({
            name: (
                (
                    OfficialMiTARouteAttention(
                        hidden_channels,
                        num_heads=heads,
                        router_count=router_count,
                        top_k=top_k,
                        dropout=dropout,
                    )
                    if attention_impl == "official"
                    else MixtureTopKAttention(
                        hidden_channels,
                        num_heads=heads,
                        router_count=router_count,
                        top_k=top_k,
                        dropout=dropout,
                    )
                )
                if self.attention_type == "mita"
                else nn.MultiheadAttention(
                    hidden_channels,
                    heads,
                    dropout=float(dropout),
                    batch_first=True,
                )
            )
            for name in self.stream_names
        })
        self.last_attention_diagnostics: dict[str, object] = {}
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, 1),
                )
                for name in self.stream_names
            }
        )
        self.topology_norm = nn.LayerNorm(hidden_channels)
        self.output = nn.Sequential(
            nn.Conv2d(
                hidden_channels * len(self.stream_names),
                hidden_channels,
                1,
            ),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )
        self.output_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        topology_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden.ndim != 4 or condition.ndim != 4:
            raise ValueError("hidden and condition must be [B,C,H,W]")
        if condition.shape[-2:] != hidden.shape[-2:]:
            condition = nn.functional.interpolate(
                condition,
                size=hidden.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        stream_tokens = {
            name: projection(hidden).flatten(2).transpose(1, 2)
            for name, projection in self.projections.items()
        }
        memory = torch.cat([*stream_tokens.values(), condition.flatten(2).transpose(1, 2)], dim=1)
        if topology_tokens is not None:
            if topology_tokens.shape[-1] != self.hidden_channels:
                raise ValueError("topology_tokens must have hidden_channels width")
            memory = torch.cat([memory, self.topology_norm(topology_tokens)], dim=1)
        gates = []
        updated = {}
        for name, tokens in stream_tokens.items():
            if self.attention_type == "mita":
                context, attention_diagnostics = self.attention[name](tokens, memory)
                self.last_attention_diagnostics[name] = attention_diagnostics
            else:
                context = self.attention[name](
                    tokens,
                    memory,
                    memory,
                    need_weights=False,
                )[0]
            gate = torch.sigmoid(self.gates[name](torch.cat([tokens, context], dim=-1)))
            updated[name] = tokens + gate * context
            gates.append(gate)
        gate_map = torch.softmax(torch.cat(gates, dim=-1), dim=-1)
        mixed = torch.cat(
            [
                updated[name].transpose(1, 2).reshape(
                    hidden.shape[0],
                    self.hidden_channels,
                    *hidden.shape[-2:],
                )
                for name in self.stream_names
            ],
            dim=1,
        )
        output = hidden + self.output_scale.to(hidden.dtype) * self.output(mixed)
        diagnostics = stream_gate_diagnostics(gate_map)
        diagnostics["stream_gates"] = gate_map.mean(dim=1)
        if self.attention_type == "mita":
            for metric in (
                "routing_entropy",
                "attention_entropy",
                "effective_expert_fraction",
            ):
                diagnostics[f"mita_{metric}"] = torch.stack(
                    [
                        self.last_attention_diagnostics[name][metric]
                        for name in self.stream_names
                    ]
                ).mean()
        return output, diagnostics


__all__ = ["SemanticFamilyStreamMixer"]

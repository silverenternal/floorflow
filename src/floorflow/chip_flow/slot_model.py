"""Global-latent chip flow with a permutation-equivariant slot decoder."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.graph_conditioning import (
    GraphAttentionBias,
    GraphMessageResidual,
)
from floorflow.chip_flow.model import BlockLayer


class GlobalSlotBlock(nn.Module):
    """Self-attention followed by global-context injection."""

    def __init__(self, hidden: int, heads: int, global_dim: int) -> None:
        super().__init__()
        self.block = BlockLayer(hidden, heads, None)
        self.context = nn.Linear(global_dim, hidden)

    def forward(
        self,
        tokens: torch.Tensor,
        global_state: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.context(global_state)[:, None]
        return self.block(tokens, bias) + context


class GlobalSetEncoder(nn.Module):
    """Encode a conditioned token set into one permutation-invariant vector."""

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        global_dim: int,
        layers: int,
        heads: int,
        graph_messages: bool = False,
        global_pool_stats: bool = False,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.layers = nn.ModuleList(
            [BlockLayer(hidden, heads, None) for _ in range(layers)]
        )
        self.graph_layers = nn.ModuleList(
            [GraphMessageResidual(hidden) for _ in range(layers)]
            if graph_messages
            else []
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, global_dim),
            nn.SiLU(),
            nn.Linear(global_dim, global_dim),
        )
        self.raw_pool = (
            nn.Parameter(torch.zeros(global_dim, input_dim * 2))
            if global_pool_stats
            else None
        )

    def forward(
        self,
        inputs: torch.Tensor,
        bias: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.input(inputs)
        for index, layer in enumerate(self.layers):
            if len(self.graph_layers):
                tokens = self.graph_layers[index](tokens, adjacency)
            tokens = layer(tokens, bias)
        pooled = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
        global_state = self.pool(pooled)
        if self.raw_pool is not None:
            raw_stats = torch.cat(
                [
                    inputs.mean(dim=1),
                    inputs.std(dim=1, correction=0),
                ],
                dim=-1,
            )
            global_state = global_state + torch.nn.functional.linear(
                raw_stats,
                self.raw_pool,
            )
        return global_state


class SlotDecoder(nn.Module):
    """Decode all token latents from a global state without particle transport."""

    def __init__(
        self,
        feature_dim: int,
        hidden: int,
        global_dim: int,
        layers: int,
        heads: int,
        graph_messages: bool = False,
        global_film: bool = False,
    ) -> None:
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.layers = nn.ModuleList(
            [
                GlobalSlotBlock(hidden, heads, global_dim * 2)
                for _ in range(layers)
            ]
        )
        self.graph_layers = nn.ModuleList(
            [GraphMessageResidual(hidden) for _ in range(layers)]
            if graph_messages
            else []
        )
        self.norm = nn.LayerNorm(hidden)
        self.global_film = (
            nn.Parameter(torch.zeros(hidden, global_dim))
            if global_film
            else None
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(
        self,
        global_state: torch.Tensor,
        condition: torch.Tensor,
        features: torch.Tensor,
        bias: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.query(features)
        context = torch.cat([global_state, condition], dim=-1)
        for index, layer in enumerate(self.layers):
            if len(self.graph_layers):
                tokens = self.graph_layers[index](tokens, adjacency)
            tokens = layer(tokens, context, bias)
        if self.global_film is not None:
            scale = torch.tanh(
                torch.nn.functional.linear(global_state, self.global_film)
            )
            tokens = tokens * (1.0 + scale[:, None])
        return self.head(self.norm(tokens))


class ChipSlotFlowModel(nn.Module):
    """Transport one invariant layout latent, then decode its token set."""

    def __init__(
        self,
        feature_dim: int = 10,
        hidden: int = 192,
        global_dim: int = 192,
        target_layers: int = 4,
        condition_layers: int = 2,
        decoder_layers: int = 4,
        heads: int = 6,
        graph_conditioning: bool = False,
        graph_features: bool | None = None,
        graph_bias: bool | None = None,
        graph_messages: bool | None = None,
        output_mode: str = "velocity",
        wire_dropout: float = 0.0,
        constraint_dropout: float = 0.0,
        global_pool_stats: bool = False,
        global_film: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        if output_mode not in {"velocity", "x0"}:
            raise ValueError("output_mode must be velocity or x0")
        self.output_mode = output_mode
        if not 0.0 <= wire_dropout <= 1.0:
            raise ValueError("wire_dropout must be in [0, 1]")
        if not 0.0 <= constraint_dropout <= 1.0:
            raise ValueError("constraint_dropout must be in [0, 1]")
        self.wire_dropout = float(wire_dropout)
        self.constraint_dropout = float(constraint_dropout)
        self.hidden = int(hidden)
        self.global_dim = int(global_dim)
        self.graph_features = (
            bool(graph_conditioning)
            if graph_features is None
            else bool(graph_features)
        )
        self.graph_bias_enabled = (
            bool(graph_conditioning)
            if graph_bias is None
            else bool(graph_bias)
        )
        self.graph_messages_enabled = (
            bool(graph_conditioning)
            if graph_messages is None
            else bool(graph_messages)
        )
        self.graph_input_enabled = (
            self.graph_features
            or self.graph_bias_enabled
            or self.graph_messages_enabled
        )
        self.graph_bias = (
            GraphAttentionBias() if self.graph_bias_enabled else None
        )
        self.target_encoder = GlobalSetEncoder(
            self.feature_dim + 3,
            self.hidden,
            self.global_dim,
            target_layers,
            heads,
            self.graph_messages_enabled,
            global_pool_stats,
        )
        self.condition_encoder = GlobalSetEncoder(
            self.feature_dim,
            self.hidden,
            self.global_dim,
            condition_layers,
            heads,
            self.graph_messages_enabled,
            global_pool_stats,
        )
        self.decoder = SlotDecoder(
            self.feature_dim,
            self.hidden,
            self.global_dim,
            decoder_layers,
            heads,
            self.graph_messages_enabled,
            global_film,
        )
        self.time = nn.Sequential(
            nn.Linear(1, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.velocity = nn.Sequential(
            nn.Linear(2 * self.global_dim + self.hidden, self.hidden * 2),
            nn.SiLU(),
            nn.Linear(self.hidden * 2, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.global_dim),
        )

    def _attention_bias(
        self,
        adjacency: torch.Tensor | None,
        key_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.graph_bias is None or adjacency is None:
            return None
        return self.graph_bias(adjacency.float(), key_mask)

    def encode_target(
        self,
        target_latent: torch.Tensor,
        features: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.target_encoder(
            torch.cat([target_latent.float(), features.float()], dim=-1),
            bias=self._attention_bias(adjacency, key_mask),
            adjacency=adjacency,
        )

    def encode_condition(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.condition_encoder(
            features.float(),
            bias=self._attention_bias(adjacency, key_mask),
            adjacency=adjacency,
        )

    def decode(
        self,
        global_state: torch.Tensor,
        condition: torch.Tensor,
        features: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(
            global_state.float(),
            condition.float(),
            features.float(),
            self._attention_bias(adjacency, key_mask),
            adjacency,
        )

    def forward(
        self,
        global_state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        time_feature = self.time(time[:, None].float())
        inputs = torch.cat([global_state, condition, time_feature], dim=-1)
        return self.velocity(inputs)

    @torch.no_grad()
    def sample(
        self,
        features: torch.Tensor,
        sampling_steps: int,
        *,
        adjacency: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        condition = self.encode_condition(features, adjacency, key_mask)
        state = torch.randn(
            condition.shape[0],
            self.global_dim,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        for step in range(sampling_steps):
            time = torch.full(
                (state.shape[0],),
                step / sampling_steps,
                device=state.device,
                dtype=state.dtype,
            )
            prediction = self(state, time, condition)
            if self.output_mode == "x0":
                velocity = (prediction - state) / (
                    1.0 - time[:, None]
                ).clamp_min(1.0e-3)
            else:
                velocity = prediction
            state = state + velocity / sampling_steps
        return self.decode(
            state,
            condition,
            features,
            adjacency,
            key_mask,
        )


__all__ = [
    "ChipSlotFlowModel",
    "GlobalSetEncoder",
    "GlobalSlotBlock",
    "SlotDecoder",
]

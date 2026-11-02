"""Masked multi-slot flow built on the existing sparse chip hierarchy."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.model import BlockLayer
from floorflow.chip_flow.sparse_flow_model import (
    SparseConditionContext,
    SparseHierarchyDecoder,
    sparse_regularization_terms,
)
from floorflow.chip_flow.sparse_hierarchy import (
    SparseHierarchyEncoder,
    SparseHierarchyEncoding,
)
from floorflow.chip_flow.sparse_layers import GraphMessageBlock


class SlotTransportVelocity(nn.Module):
    """Predict velocity over global and parent-slot tokens."""

    def __init__(
        self,
        dim: int,
        *,
        state_tokens: int,
        hidden: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(dim * state_tokens, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.input = nn.Sequential(
            nn.Linear(dim * 2 + hidden, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.layers = nn.ModuleList(
            [BlockLayer(dim, heads, None) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        condition_token = self.condition(condition)
        time_token = self.time(time[:, None])
        inputs = torch.cat(
            [
                state,
                condition_token[:, None].expand_as(state),
                time_token[:, None].expand_as(state),
            ],
            dim=-1,
        )
        tokens = state + self.input(inputs)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.head(self.norm(tokens))


class ChipSlotTransportFlowModel(nn.Module):
    """Flow a global token plus learned parent slots before block decoding."""

    def __init__(
        self,
        feature_dim: int = 10,
        hidden: int = 192,
        global_dim: int = 192,
        target_layers: int = 4,
        condition_layers: int = 2,
        decoder_layers: int = 4,
        heads: int = 6,
        max_slots: int = 8,
        min_slots: int = 2,
        assignment_top_k: int = 2,
        assignment_temperature: float = 1.0,
        straight_through_assignment: bool = True,
        decoder_top_k: int = 4,
        transport_layers: int = 2,
        residual_slots: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.global_dim = int(global_dim)
        self.max_slots = int(max_slots)
        self.residual_slots = bool(residual_slots)
        self.state_tokens = self.max_slots + 1
        self.condition_hierarchy = SparseHierarchyEncoder(
            self.feature_dim,
            hidden=self.global_dim,
            layers=condition_layers,
            heads=heads,
            max_slots=max_slots,
            min_slots=min_slots,
            assignment_top_k=assignment_top_k,
            assignment_temperature=assignment_temperature,
            straight_through_assignment=straight_through_assignment,
        )
        self.target_hierarchy = SparseHierarchyEncoder(
            self.feature_dim + 3,
            hidden=self.global_dim,
            layers=target_layers,
            heads=heads,
            max_slots=max_slots,
            min_slots=min_slots,
            assignment_top_k=assignment_top_k,
            assignment_temperature=assignment_temperature,
            straight_through_assignment=straight_through_assignment,
        )
        self.velocity = SlotTransportVelocity(
            self.global_dim,
            state_tokens=self.state_tokens,
            hidden=self.hidden,
            layers=transport_layers,
            heads=heads,
        )
        self.decoder = SparseHierarchyDecoder(
            self.feature_dim,
            self.hidden,
            self.global_dim,
            decoder_layers,
            heads,
            max_slots,
            decoder_top_k,
        )
        self.latent_graph = GraphMessageBlock(3)

    @staticmethod
    def condition_vector(context: SparseConditionContext) -> torch.Tensor:
        hierarchy = context.hierarchy
        slots = hierarchy.slot_tokens * hierarchy.active_slot_mask[:, :, None]
        state = torch.cat(
            [hierarchy.global_state[:, None], slots],
            dim=1,
        )
        return state.flatten(1)

    @staticmethod
    def state_mask(context: SparseConditionContext) -> torch.Tensor:
        hierarchy = context.hierarchy
        return torch.cat(
            [
                torch.ones(
                    hierarchy.active_slot_mask.shape[0],
                    1,
                    device=hierarchy.active_slot_mask.device,
                    dtype=torch.bool,
                ),
                hierarchy.active_slot_mask,
            ],
            dim=1,
        )

    def encode_condition_context(
        self,
        features: torch.Tensor,
        constraints: torch.Tensor | None = None,
        *,
        b2b: torch.Tensor | None = None,
        p2b: torch.Tensor | None = None,
        pins: torch.Tensor | None = None,
        n_b2b: torch.Tensor | None = None,
        n_p2b: torch.Tensor | None = None,
        n_pins: torch.Tensor | None = None,
    ) -> SparseConditionContext:
        hierarchy = self.condition_hierarchy(
            features,
            constraints,
            b2b=b2b,
            p2b=p2b,
            pins=pins,
            n_b2b=n_b2b,
            n_p2b=n_p2b,
            n_pins=n_pins,
        )
        return SparseConditionContext(hierarchy, hierarchy.global_state)

    def encode_target(
        self,
        target_latent: torch.Tensor,
        features: torch.Tensor,
        constraints: torch.Tensor | None = None,
        *,
        b2b: torch.Tensor | None = None,
        p2b: torch.Tensor | None = None,
        pins: torch.Tensor | None = None,
        n_b2b: torch.Tensor | None = None,
        n_p2b: torch.Tensor | None = None,
        n_pins: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_features = torch.cat(
            [target_latent.float(), features.float()],
            dim=-1,
        )
        hierarchy = self.target_hierarchy(
            target_features,
            constraints,
            b2b=b2b,
            p2b=p2b,
            pins=pins,
            n_b2b=n_b2b,
            n_p2b=n_p2b,
            n_pins=n_pins,
        )
        slots = hierarchy.slot_tokens * hierarchy.active_slot_mask[:, :, None]
        return torch.cat(
            [hierarchy.global_state[:, None], slots],
            dim=1,
        )

    def to_residual_target(
        self,
        target_state: torch.Tensor,
        context: SparseConditionContext,
    ) -> torch.Tensor:
        if not self.residual_slots:
            return target_state
        active = context.hierarchy.active_slot_mask
        slots = (
            target_state[:, 1:]
            - context.hierarchy.slot_tokens
        ) * active[:, :, None]
        return torch.cat([target_state[:, :1], slots], dim=1)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity(
            state.float(),
            time.float(),
            condition.float(),
        )

    def sparse_regularization(
        self,
        context: SparseConditionContext,
        decode_diagnostics: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return sparse_regularization_terms(
            context,
            decode_diagnostics,
            self.max_slots,
        )

    def decode_transport(
        self,
        state: torch.Tensor,
        context: SparseConditionContext,
        features: torch.Tensor,
        constraints: torch.Tensor | None = None,
        *,
        b2b: torch.Tensor | None = None,
        p2b: torch.Tensor | None = None,
        pins: torch.Tensor | None = None,
        n_b2b: torch.Tensor | None = None,
        n_p2b: torch.Tensor | None = None,
        n_pins: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del constraints
        generated_slots = state[:, 1:]
        if self.residual_slots:
            generated_slots = (
                context.hierarchy.slot_tokens + generated_slots
            )
        hierarchy = SparseHierarchyEncoding(
            tokens=context.hierarchy.tokens,
            slot_tokens=(
                generated_slots
                * context.hierarchy.active_slot_mask[:, :, None]
            ),
            slot_weights=context.hierarchy.slot_weights,
            active_slot_mask=context.hierarchy.active_slot_mask,
            token_assignments=context.hierarchy.token_assignments,
            global_state=state[:, 0],
            diagnostics=context.hierarchy.diagnostics,
        )
        latent, diagnostics = self.decoder(
            state[:, 0],
            context.global_state,
            features,
            hierarchy,
        )
        latent = self.latent_graph(
            latent,
            b2b=b2b,
            p2b=p2b,
            pins=pins,
            n_b2b=n_b2b,
            n_p2b=n_p2b,
            n_pins=n_pins,
        )
        return latent, diagnostics

    @torch.no_grad()
    def sample(
        self,
        features: torch.Tensor,
        sampling_steps: int,
        constraints: torch.Tensor | None = None,
        *,
        b2b: torch.Tensor | None = None,
        p2b: torch.Tensor | None = None,
        pins: torch.Tensor | None = None,
        n_b2b: torch.Tensor | None = None,
        n_p2b: torch.Tensor | None = None,
        n_pins: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context = self.encode_condition_context(
            features,
            constraints,
            b2b=b2b,
            p2b=p2b,
            pins=pins,
            n_b2b=n_b2b,
            n_p2b=n_p2b,
            n_pins=n_pins,
        )
        condition = self.condition_vector(context)
        state = torch.randn(
            features.shape[0],
            self.state_tokens,
            self.global_dim,
            device=features.device,
            dtype=features.dtype,
            generator=generator,
        )
        mask = self.state_mask(context)
        state = state * mask[:, :, None]
        if self.residual_slots:
            state[:, 1:] = 0.0
        for step in range(sampling_steps):
            time = torch.full(
                (features.shape[0],),
                step / sampling_steps,
                device=features.device,
                dtype=features.dtype,
            )
            velocity = self(state, time, condition) * mask[:, :, None]
            state = state + velocity / sampling_steps
        return self.decode_transport(
            state,
            context,
            features,
            b2b=b2b,
            p2b=p2b,
            pins=pins,
            n_b2b=n_b2b,
            n_p2b=n_p2b,
            n_pins=n_pins,
        )


__all__ = ["ChipSlotTransportFlowModel", "SlotTransportVelocity"]

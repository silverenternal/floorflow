"""Typed netlist-aware graph conditioning for chip trajectories."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.complexity_conditioning import (
    ComplexityConditioning,
    case_complexity_features,
)
from floorflow.chip_flow.degree_conditioned_expert import (
    DegreeConditionedExpert,
)
from floorflow.chip_flow.global_spec import GlobalSpecConditioner
from floorflow.chip_flow.local_geometry_features import (
    RawLocalGeometryEncoder,
    raw_local_geometry_features,
)
from floorflow.chip_flow.relation_gate_balance import (
    relation_gate_regularization,
)
from floorflow.chip_flow.semantic_conditioning import MaskedSemanticEncoder
from floorflow.chip_flow.sparse_layers import GraphMessageBlock, masked_token_mean
from floorflow.chip_flow.typed_channel_fusion import fuse_typed_channels
from floorflow.chip_flow.typed_channel_state import (
    TypedChannelState,
    TypedGraphOutput,
)
from floorflow.chip_flow.typed_cross_attention import TypedCrossAttention


def _pool_groups(
    tokens: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
    block_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, hidden = tokens.shape
    pooled = tokens.new_zeros(batch, group_count, hidden)
    valid = torch.zeros(
        batch,
        group_count,
        device=tokens.device,
        dtype=torch.bool,
    )
    for group in range(1, group_count + 1):
        selected = (group_ids == group) & block_mask
        valid[:, group - 1] = selected.any(dim=1)
        pooled[:, group - 1] = masked_token_mean(tokens, selected)
    return pooled, valid


def _boundary_masks(constraints: torch.Tensor) -> torch.Tensor:
    codes = constraints[..., 4].long()
    return torch.stack(
        [(codes & bit).ne(0) for bit in (1, 2, 4, 8)],
        dim=-1,
    )


class TypedGraphEncoder(nn.Module):
    """Encode block, structural, constraint and port channels with typed edges."""

    def __init__(
        self,
        feature_dim: int = 19,
        hidden: int = 192,
        heads: int = 6,
        graph_layers: int = 2,
        cluster_slots: int = 4,
        mib_slots: int = 1,
        typed_mode: str = "typed",
        edge_gate: bool = False,
        symmetric_b2b_messages: bool = False,
        degree_expert_conditioning: bool = False,
        relation_specific: bool = False,
        incidence_normalized_relations: bool = False,
        complexity_conditioning: bool = False,
        global_spec_conditioning: bool = False,
        node_semantic_dim: int = 0,
        port_semantic_dim: int = 0,
    ) -> None:
        super().__init__()
        if feature_dim != 19:
            raise ValueError("typed encoder requires the 19-column chip schema")
        if typed_mode not in (
            "typed",
            "typed_bias",
            "dense",
            "no_structural",
            "no_constraint",
        ):
            raise ValueError(f"unknown typed attention mode: {typed_mode}")
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.typed_mode = typed_mode
        self.cluster_slots = int(cluster_slots)
        self.mib_slots = int(mib_slots)
        self.block_input = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.graph_layers = nn.ModuleList(
            [
                GraphMessageBlock(
                    hidden,
                    edge_gate=edge_gate,
                    symmetric_edges=symmetric_b2b_messages,
                )
                for _ in range(graph_layers)
            ]
        )
        self.degree_expert = (
            DegreeConditionedExpert(hidden)
            if degree_expert_conditioning
            else None
        )
        self.port_input = nn.Linear(3, hidden)
        self.structural_type = nn.Parameter(
            torch.randn(self.cluster_slots + self.mib_slots + 4, hidden) * 0.02
        )
        self.constraint_type = nn.Parameter(torch.randn(3, hidden) * 0.02)
        self.port_type = nn.Parameter(torch.randn(hidden) * 0.02)
        self.global_to_structure = nn.Linear(hidden, hidden)
        self.constraint_to_port = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.global_to_structure.weight)
        nn.init.zeros_(self.global_to_structure.bias)
        nn.init.zeros_(self.constraint_to_port.weight)
        nn.init.zeros_(self.constraint_to_port.bias)
        self.cross_attention = TypedCrossAttention(
            hidden,
            heads,
            relation_specific=relation_specific,
            incidence_normalized_relations=(
                incidence_normalized_relations
            ),
        )
        self.relation_specific = bool(relation_specific)
        self.incidence_normalized_relations = bool(
            incidence_normalized_relations
        )
        self.complexity_conditioning = bool(complexity_conditioning)
        self.global_spec_conditioning = bool(global_spec_conditioning)
        self.node_semantic_dim = int(node_semantic_dim)
        self.port_semantic_dim = int(port_semantic_dim)
        self.complexity = (
            ComplexityConditioning(hidden)
            if self.complexity_conditioning
            else None
        )
        self.global_spec_conditioner = (
            GlobalSpecConditioner(hidden)
            if global_spec_conditioning
            else None
        )
        self.node_semantic_encoder = (
            MaskedSemanticEncoder(node_semantic_dim, hidden)
            if node_semantic_dim > 0
            else None
        )
        self.port_semantic_encoder = (
            MaskedSemanticEncoder(port_semantic_dim, hidden)
            if port_semantic_dim > 0
            else None
        )
        self.local_geometry_encoder = RawLocalGeometryEncoder(hidden)
        self.norm = nn.LayerNorm(hidden)

    def _port_tokens(
        self,
        pins: torch.Tensor | None,
        p2b: torch.Tensor | None,
        n_p2b: torch.Tensor | None,
        n_pins: torch.Tensor | None,
        areas: torch.Tensor,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = areas.shape[0]
        if pins is None:
            empty = areas.new_zeros(batch, 1, self.hidden)
            mask = torch.zeros(batch, 1, device=areas.device, dtype=torch.bool)
            incidence = torch.zeros(
                batch,
                count,
                1,
                device=areas.device,
                dtype=torch.bool,
            )
            return empty, mask, incidence
        pin_count = pins.shape[1]
        rows = torch.arange(pin_count, device=pins.device)[None]
        pin_mask = (
            rows < n_pins.to(pins.device).long()[:, None]
            if n_pins is not None
            else torch.ones(batch, pin_count, device=pins.device, dtype=torch.bool)
        )
        scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
        pin_weight = pins.new_zeros(batch, pin_count)
        incidence = torch.zeros(
            batch,
            count,
            pin_count,
            device=areas.device,
            dtype=torch.bool,
        )
        if p2b is not None and p2b.shape[1]:
            edge_rows = torch.arange(p2b.shape[1], device=p2b.device)[None]
            edge_valid = (
                edge_rows < n_p2b.to(p2b.device).long()[:, None]
                if n_p2b is not None
                else torch.ones(
                    batch,
                    p2b.shape[1],
                    device=p2b.device,
                    dtype=torch.bool,
                )
            )
            pin_index = p2b[..., 0].long()
            block_index = p2b[..., 1].long()
            valid = (
                edge_valid
                & (pin_index >= 0)
                & (pin_index < pin_count)
                & (block_index >= 0)
                & (block_index < count)
            )
            batch_index = torch.arange(batch, device=p2b.device)[:, None]
            flat_incidence = incidence.view(-1)
            flat_index = (
                (batch_index * count + block_index) * pin_count + pin_index
            )
            flat_incidence[flat_index[valid]] = True
            flat_weight = pin_weight.view(-1)
            flat_weight.index_add_(
                0,
                (batch_index * pin_count + pin_index)[valid].reshape(-1),
                p2b[..., 2][valid].to(pin_weight.dtype),
            )
        features = torch.cat(
            [
                pins.float() / scale[:, None, None],
                pin_weight.log1p()[..., None],
            ],
            dim=-1,
        )
        return self.port_input(features) + self.port_type, pin_mask, incidence

    def encode_channels(
        self,
        features: torch.Tensor,
        constraints: torch.Tensor,
        *,
        areas: torch.Tensor,
        b2b: torch.Tensor | None = None,
        p2b: torch.Tensor | None = None,
        pins: torch.Tensor | None = None,
        n_b2b: torch.Tensor | None = None,
        n_p2b: torch.Tensor | None = None,
        n_pins: torch.Tensor | None = None,
        global_spec: torch.Tensor | None = None,
        global_spec_mask: torch.Tensor | None = None,
        node_features: torch.Tensor | None = None,
        node_feature_mask: torch.Tensor | None = None,
        port_features: torch.Tensor | None = None,
        port_feature_mask: torch.Tensor | None = None,
    ) -> TypedChannelState:
        batch, count, _ = features.shape
        if self.global_spec_conditioner is not None:
            if global_spec is None:
                raise ValueError(
                    "global-spec conditioning requires batch['global_spec']"
                )
            if global_spec.shape != (batch, self.global_spec_conditioner.input_dim):
                raise ValueError("global spec must have shape (batch, spec_dim)")
        if self.node_semantic_encoder is not None:
            if node_features is None:
                raise ValueError(
                    "node semantic conditioning requires batch['node_features']"
                )
            expected = (batch, count, self.node_semantic_dim)
            if node_features.shape != expected:
                raise ValueError(
                    f"node semantic features must have shape {expected}"
                )
        block_mask = torch.ones(
            batch,
            count,
            device=features.device,
            dtype=torch.bool,
        )
        blocks = self.block_input(features.float())
        if self.node_semantic_encoder is not None and node_features is not None:
            blocks = blocks + self.node_semantic_encoder(
                node_features,
                node_feature_mask,
            )
        local_geometry = self.local_geometry_encoder(
            raw_local_geometry_features(
                areas,
                b2b,
                p2b,
                pins,
                n_b2b,
                n_p2b,
                n_pins,
            )
        )
        blocks = blocks + local_geometry
        for layer in self.graph_layers:
            blocks = layer(
                blocks,
                b2b=b2b,
                p2b=p2b,
                pins=pins,
                n_b2b=n_b2b,
                n_p2b=n_p2b,
                n_pins=n_pins,
            )
        extra_diagnostics = {}
        if self.degree_expert is not None:
            blocks, extra_diagnostics = self.degree_expert(
                blocks,
                b2b=b2b,
                p2b=p2b,
                n_b2b=n_b2b,
                n_p2b=n_p2b,
            )
        global_token = masked_token_mean(blocks, block_mask)[:, None]
        if self.global_spec_conditioner is not None:
            global_token = global_token + self.global_spec_conditioner(
                global_spec,
                global_spec_mask,
            )[:, None]
        if self.complexity is not None:
            complexity = case_complexity_features(
                areas,
                constraints,
                b2b,
                p2b,
                pins,
                n_b2b,
                n_p2b,
                n_pins,
            )
            global_token = global_token + self.complexity(complexity)[:, None]
        else:
            complexity = None
        cluster_tokens, cluster_mask = _pool_groups(
            blocks,
            constraints[..., 3].long(),
            self.cluster_slots,
            block_mask,
        )
        mib_tokens, mib_mask = _pool_groups(
            blocks,
            constraints[..., 2].long(),
            self.mib_slots,
            block_mask,
        )
        boundaries = _boundary_masks(constraints)
        boundary_tokens = torch.stack(
            [
                masked_token_mean(blocks, boundaries[..., index])
                for index in range(4)
            ],
            dim=1,
        )
        boundary_mask = boundaries.any(dim=1)
        structural = torch.cat(
            [cluster_tokens, mib_tokens, boundary_tokens],
            dim=1,
        )
        structural_mask = torch.cat(
            [cluster_mask, mib_mask, boundary_mask],
            dim=1,
        )
        if self.typed_mode == "no_structural":
            structural_mask = torch.zeros_like(structural_mask)
        structural = (
            structural
            + self.structural_type
            + self.global_to_structure(global_token)
        )
        anchored = constraints[..., :2].ne(0).any(dim=-1)
        mib_cluster = constraints[..., 2:4].ne(0).any(dim=-1)
        constraint_tokens = torch.stack(
            [
                masked_token_mean(blocks, anchored),
                masked_token_mean(blocks, boundaries.any(dim=-1)),
                masked_token_mean(blocks, mib_cluster),
            ],
            dim=1,
        )
        constraint_mask = torch.stack(
            [
                anchored.any(dim=1),
                boundaries.any(dim=(1, 2)),
                mib_cluster.any(dim=1),
            ],
            dim=1,
        )
        constraint_tokens = constraint_tokens + self.constraint_type
        if self.typed_mode == "no_constraint":
            constraint_mask = torch.zeros_like(constraint_mask)
        ports, _port_mask, incidence = self._port_tokens(
            pins,
            p2b,
            n_p2b,
            n_pins,
            areas,
            count,
        )
        if self.port_semantic_encoder is not None:
            if port_features is None:
                raise ValueError(
                    "port semantic conditioning requires batch['port_features']"
                )
            expected = (batch, ports.shape[1], self.port_semantic_dim)
            if port_features.shape != expected:
                raise ValueError(
                    f"port semantic features must have shape {expected}"
                )
            ports = ports + self.port_semantic_encoder(
                port_features,
                port_feature_mask,
            )
        port_context = masked_token_mean(
            constraint_tokens,
            constraint_mask,
        )[:, None]
        ports = ports + self.constraint_to_port(port_context)
        memory = torch.cat(
            [global_token, structural, constraint_tokens, ports],
            dim=1,
        )
        global_allowed = torch.ones(
            batch,
            count,
            1,
            device=features.device,
            dtype=torch.bool,
        )
        structural_allowed = structural_mask[:, None].expand(
            batch,
            count,
            self.cluster_slots + self.mib_slots + 4,
        )
        constraint_allowed = constraint_mask[:, None].expand(
            batch,
            count,
            3,
        )
        if self.typed_mode == "dense":
            allowed = torch.ones(
                batch,
                count,
                memory.shape[1],
                device=features.device,
                dtype=torch.bool,
            )
        else:
            allowed = torch.cat(
                [
                    global_allowed,
                    structural_allowed,
                    constraint_allowed,
                    incidence,
                ],
                dim=-1,
            )
        group_sizes = (
            (
                1,
                self.cluster_slots + self.mib_slots + 4,
                3,
                incidence.shape[-1],
            )
            if self.typed_mode == "typed_bias"
            else None
        )
        relation_memories = None
        relation_masks = None
        if self.relation_specific:
            structural_query_mask = torch.cat(
                [
                    constraints[..., 3].long()[:, :, None].eq(
                        torch.arange(
                            1,
                            self.cluster_slots + 1,
                            device=features.device,
                        )[None, None]
                    ),
                    constraints[..., 2].long()[:, :, None].eq(1),
                    boundaries,
                ],
                dim=-1,
            )
            structural_query_mask = (
                structural_query_mask & structural_mask[:, None]
            )
            constraint_query_mask = torch.cat(
                [
                    anchored[..., None],
                    boundaries.any(dim=-1)[..., None],
                    mib_cluster[..., None],
                ],
                dim=-1,
            ) & constraint_mask[:, None]
            relation_memories = (global_token, structural, constraint_tokens, ports)
            relation_masks = (global_allowed, structural_query_mask, constraint_query_mask, incidence)
        diagnostics = {
            "typed_edge_fraction": allowed.float().mean(),
            "port_edge_fraction": incidence.float().mean(),
            "valid_structural_tokens": structural_mask.float().sum(),
            "valid_constraint_tokens": constraint_mask.float().sum(),
            "local_geometry_residual_norm": local_geometry.detach().norm(dim=-1).mean(),
            **extra_diagnostics,
        }
        if complexity is not None:
            diagnostics["complexity_feature_norm"] = complexity.norm(dim=-1).mean()
        return TypedChannelState(
            block_tokens=blocks,
            global_token=global_token,
            structural_tokens=structural,
            constraint_tokens=constraint_tokens,
            port_tokens=ports,
            structural_mask=structural_mask,
            constraint_mask=constraint_mask,
            port_mask=_port_mask,
            port_block_incidence=incidence,
            allowed=allowed,
            group_sizes=group_sizes,
            relation_memories=relation_memories,
            relation_masks=relation_masks,
            diagnostics=diagnostics,
        )

    def fuse_channels(self, state: TypedChannelState) -> TypedGraphOutput:
        """Train or run cross-attention independently from channel encoding."""

        return fuse_typed_channels(
            self.cross_attention,
            self.norm,
            state,
            relation_specific=self.relation_specific,
        )

    def forward(
        self,
        features: torch.Tensor,
        constraints: torch.Tensor,
        *,
        areas: torch.Tensor,
        b2b: torch.Tensor | None = None,
        p2b: torch.Tensor | None = None,
        pins: torch.Tensor | None = None,
        n_b2b: torch.Tensor | None = None,
        n_p2b: torch.Tensor | None = None,
        n_pins: torch.Tensor | None = None,
        global_spec: torch.Tensor | None = None,
        global_spec_mask: torch.Tensor | None = None,
        node_features: torch.Tensor | None = None,
        node_feature_mask: torch.Tensor | None = None,
        port_features: torch.Tensor | None = None,
        port_feature_mask: torch.Tensor | None = None,
    ) -> TypedGraphOutput:
        state = self.encode_channels(
            features,
            constraints,
            areas=areas,
            b2b=b2b,
            p2b=p2b,
            pins=pins,
            n_b2b=n_b2b,
            n_p2b=n_p2b,
            n_pins=n_pins,
            global_spec=global_spec,
            global_spec_mask=global_spec_mask,
            node_features=node_features,
            node_feature_mask=node_feature_mask,
            port_features=port_features,
            port_feature_mask=port_feature_mask,
        )
        return self.fuse_channels(state)


__all__ = [
    "TypedGraphEncoder",
    "TypedGraphOutput",
    "relation_gate_regularization",
]

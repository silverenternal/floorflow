"""Shared state containers for coupled chip hierarchy flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from trinity.models.backbone import DenoiserCond

from floorflow.chip_flow.netlist_hypergraph import NetlistHypergraphContext
from floorflow.chip_flow.typed_graph_encoder import TypedGraphOutput


@dataclass(frozen=True)
class HierarchyContext:
    typed: TypedGraphOutput
    denoiser_cond: DenoiserCond
    outline_state: torch.Tensor | None = None
    edge_state: StateEdgeContext | None = None
    order_cluster_ids: torch.Tensor | None = None
    order_cluster_mask: torch.Tensor | None = None
    coarse_membership: torch.Tensor | None = None
    block_areas: torch.Tensor | None = None
    complexity: torch.Tensor | None = None
    global_spec: torch.Tensor | None = None
    global_spec_mask: torch.Tensor | None = None
    node_features: torch.Tensor | None = None
    node_feature_mask: torch.Tensor | None = None
    port_features: torch.Tensor | None = None
    port_feature_mask: torch.Tensor | None = None
    pin_positions: torch.Tensor | None = None
    pin_mask: torch.Tensor | None = None
    netlist: NetlistHypergraphContext | None = None
    native_group_moment_state: bool = False
    constraints: torch.Tensor | None = None


@dataclass(frozen=True)
class StateEdgeContext:
    cache: dict[str, torch.Tensor]


@dataclass(frozen=True)
class HierarchyState:
    global_state: torch.Tensor
    structural_state: torch.Tensor
    block_state: torch.Tensor
    port_state: torch.Tensor
    previous_block_state: torch.Tensor | None = None
    structural_moment_state: torch.Tensor | None = None


@dataclass(frozen=True)
class HierarchyPrediction:
    global_state: torch.Tensor
    structural_state: torch.Tensor
    block_state: torch.Tensor
    port_state: torch.Tensor
    block_base_state: torch.Tensor | None = None
    projector_residual: torch.Tensor | None = None
    legalizer_displacement: torch.Tensor | None = None
    legalizer_rung_logits: torch.Tensor | None = None
    structural_relations: torch.Tensor | None = None
    structural_moment_state: torch.Tensor | None = None
    shape_mode_rho_residuals: torch.Tensor | None = None
    shape_mode_latents: torch.Tensor | None = None
    shape_mode_logits: torch.Tensor | None = None
    atom_diagnostics: dict[str, torch.Tensor] | None = None


__all__ = [
    "HierarchyContext",
    "HierarchyPrediction",
    "HierarchyState",
    "StateEdgeContext",
]

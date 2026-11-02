"""Dynamic netlist hyperedge conditioning for chip flow trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NetlistHypergraphContext:
    features: torch.Tensor
    block_incidence: torch.Tensor
    mask: torch.Tensor
    port_incidence: torch.Tensor | None = None


def netlist_context_from_batch(batch: dict) -> NetlistHypergraphContext | None:
    """Read an optional netlist hypergraph batch contract."""

    has_features = "net_features" in batch
    has_incidence = "net_block_incidence" in batch
    if has_features != has_incidence:
        raise ValueError(
            "net_features and net_block_incidence must be provided together"
        )
    if not has_features:
        return None
    features = batch["net_features"].float()
    incidence = batch["net_block_incidence"].bool()
    batch_size, count = batch["areas"].shape
    net_count = features.shape[1]
    if features.ndim != 3 or features.shape[0] != batch_size:
        raise ValueError("net_features must have shape (batch, nets, features)")
    if features.shape[-1] != 4:
        raise ValueError("net_features must have four columns")
    if incidence.shape != (batch_size, net_count, count):
        raise ValueError("net_block_incidence shape mismatch")
    mask = batch.get("net_mask")
    if mask is None:
        mask = incidence.any(dim=-1)
    else:
        mask = mask.bool()
    if mask.shape != (batch_size, net_count):
        raise ValueError("net_mask shape mismatch")
    incidence = incidence & mask[..., None]
    port_incidence = batch.get("net_port_incidence")
    if port_incidence is not None:
        port_incidence = port_incidence.bool()
        if port_incidence.shape[:2] != mask.shape:
            raise ValueError("net_port_incidence batch/net shape mismatch")
        if "pins" in batch and port_incidence.shape[-1] != batch["pins"].shape[1]:
            raise ValueError(
                "net_port_incidence columns must align with batch['pins']"
            )
        port_incidence = port_incidence & mask[..., None]
    return NetlistHypergraphContext(
        features=features,
        block_incidence=incidence,
        mask=mask,
        port_incidence=port_incidence,
    )


class NetlistHyperedgeConditioner(nn.Module):
    """Zero-gated per-step aggregation over current multi-terminal nets."""

    def __init__(
        self,
        hidden: int,
        *,
        feature_dim: int,
        width: int | None = None,
    ) -> None:
        super().__init__()
        if hidden <= 0 or feature_dim <= 0:
            raise ValueError("hidden and feature_dim must be positive")
        self.hidden = int(hidden)
        self.feature_dim = int(feature_dim)
        bottleneck = int(width or max(16, hidden // 8))
        self.feature_encoder = nn.Linear(self.feature_dim, bottleneck)
        self.state_encoder = nn.Linear(7, bottleneck)
        self.token_encoder = nn.Linear(hidden, bottleneck)
        self.port_state_encoder = nn.Linear(2, bottleneck)
        self.port_token_encoder = nn.Linear(hidden, bottleneck)
        self.network = nn.Sequential(
            nn.LayerNorm(5 * bottleneck),
            nn.Linear(5 * bottleneck, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 3),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        block_state: torch.Tensor,
        block_tokens: torch.Tensor,
        context: NetlistHypergraphContext,
        *,
        port_state: torch.Tensor | None = None,
        port_tokens: torch.Tensor | None = None,
        port_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if block_state.ndim != 3 or block_state.shape[-1] != 3:
            raise ValueError("block_state must have shape (batch, blocks, 3)")
        if block_tokens.shape[:2] != block_state.shape[:2]:
            raise ValueError("block tokens and state must share batch/block axes")
        if block_tokens.shape[-1] != self.hidden:
            raise ValueError("block token hidden size mismatch")
        if context.block_incidence.shape != (
            block_state.shape[0],
            context.features.shape[1],
            block_state.shape[1],
        ):
            raise ValueError("netlist hypergraph context shape mismatch")
        if context.features.shape != (
            block_state.shape[0],
            context.block_incidence.shape[1],
            self.feature_dim,
        ):
            raise ValueError("netlist hypergraph feature shape mismatch")
        if context.mask.shape != context.block_incidence.shape[:2]:
            raise ValueError("netlist hypergraph mask shape mismatch")
        has_port_inputs = (port_state is not None, port_tokens is not None, port_mask is not None)
        if any(has_port_inputs) and not all(has_port_inputs):
            raise ValueError("port_state, port_tokens and port_mask must be provided together")
        valid_nets = context.mask.bool()
        safe_features = torch.where(
            valid_nets[..., None],
            context.features.float(),
            context.features.new_zeros(()),
        )
        incidence = (
            context.block_incidence.float()
            * valid_nets.float()[..., None]
        )
        count = incidence.sum(dim=-1).clamp_min(1.0)
        pooled_state = torch.bmm(incidence, block_state.float()) / count[..., None]
        pooled_tokens = torch.bmm(incidence, block_tokens.float()) / count[..., None]
        positions = block_state.float()[..., :2]
        lower = torch.full_like(positions, float("inf")).unsqueeze(1)
        upper = torch.full_like(positions, float("-inf")).unsqueeze(1)
        selected = incidence[..., None] * positions[:, None]
        lower = torch.where(
            incidence[..., None].bool(),
            selected,
            lower,
        ).amin(dim=2)
        upper = torch.where(
            incidence[..., None].bool(),
            selected,
            upper,
        ).amax(dim=2)
        span = (upper - lower).masked_fill(~valid_nets[..., None], 0.0)
        if context.port_incidence is not None:
            if context.port_incidence.shape[:2] != valid_nets.shape:
                raise ValueError("netlist port incidence batch/net shape mismatch")
            port_valid = context.port_incidence.bool()
            if port_mask is not None:
                if port_mask.shape != (
                    block_tokens.shape[0],
                    port_valid.shape[-1],
                ):
                    raise ValueError("port_mask shape does not match net_port_incidence")
                port_valid = port_valid & port_mask[:, None, :].bool()
            port_count = (
                port_valid & valid_nets[..., None]
            ).sum(dim=-1).float().log1p()
        else:
            port_count = torch.zeros_like(count)
        zero_port_state = block_tokens.new_zeros(
            block_tokens.shape[0],
            context.features.shape[1],
            2,
        )
        zero_port_tokens = block_tokens.new_zeros(
            block_tokens.shape[0],
            context.features.shape[1],
            block_tokens.shape[-1],
        )
        if port_state is not None and context.port_incidence is not None:
            expected_ports = context.port_incidence.shape[-1]
            if port_state.shape != (block_tokens.shape[0], expected_ports, 2):
                raise ValueError("port_state shape does not match net_port_incidence")
            if port_tokens.shape != (
                block_tokens.shape[0],
                expected_ports,
                block_tokens.shape[-1],
            ):
                raise ValueError("port_tokens shape does not match net_port_incidence")
            if port_mask.shape != (block_tokens.shape[0], expected_ports):
                raise ValueError("port_mask shape does not match net_port_incidence")
            port_weights = (
                context.port_incidence.bool()
                & valid_nets[..., None]
                & port_mask[:, None, :].bool()
            )
            port_mass = port_weights.sum(dim=-1).clamp_min(1.0)
            zero_port_state = torch.bmm(
                port_weights.float(),
                port_state.float(),
            ) / port_mass[..., None]
            zero_port_tokens = torch.bmm(
                port_weights.float(),
                port_tokens.float(),
            ) / port_mass[..., None]
        dynamic_state = torch.cat(
            [
                pooled_state,
                span,
                count.log1p()[..., None],
                port_count[..., None],
            ],
            dim=-1,
        )
        messages = self.network(
            torch.cat(
                [
                    self.feature_encoder(safe_features),
                    self.state_encoder(dynamic_state),
                    self.token_encoder(pooled_tokens),
                    self.port_state_encoder(zero_port_state),
                    self.port_token_encoder(zero_port_tokens),
                ],
                dim=-1,
            )
        )
        degree = incidence.sum(dim=1).clamp_min(1.0)[..., None]
        block_message = torch.bmm(incidence.transpose(1, 2), messages) / degree
        return self.gate * block_message


__all__ = [
    "NetlistHyperedgeConditioner",
    "NetlistHypergraphContext",
    "netlist_context_from_batch",
]

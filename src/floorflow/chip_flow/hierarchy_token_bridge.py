"""Trajectory-aware L0/L1 conditioning for the Trinity token backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from trinity.models.backbone import DenoiserCond, SetTransformerDenoiser
from trinity.models.presets import DenoiserArchConfig

from floorflow.chip_flow.typed_graph_encoder import TypedCrossAttention


@dataclass(frozen=True)
class HierarchyTokenBridgeInput:
    """Per-step coarse hierarchy predictions consumed by the backbone."""

    block_tokens: torch.Tensor
    block_state: torch.Tensor
    global_prediction: torch.Tensor
    structural_prediction: torch.Tensor
    structural_tokens: torch.Tensor
    structural_mask: torch.Tensor
    time: torch.Tensor
    membership: torch.Tensor | None = None


class HierarchyTokenBridge(nn.Module):
    """Fuse predicted L0/L1 states into hidden L2 tokens before the backbone."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        *,
        bounded: bool = False,
        structural_only: bool = False,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.block_norm = nn.LayerNorm(hidden)
        self.state_encoder = nn.Linear(3, hidden)
        self.time_encoder = nn.Linear(1, hidden)
        self.global_encoder = nn.Linear(4, hidden)
        self.structural_encoder = nn.Linear(3, hidden)
        self.cross = TypedCrossAttention(hidden, heads)
        bottleneck = max(32, hidden // 2)
        self.update = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden),
        )
        self.bounded = bool(bounded)
        self.structural_only = bool(structural_only)
        if self.bounded:
            nn.init.normal_(self.update[-1].weight, std=0.02)
            nn.init.zeros_(self.update[-1].bias)
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            nn.init.zeros_(self.update[-1].weight)
            nn.init.zeros_(self.update[-1].bias)
            self.gate = nn.Parameter(torch.ones(1))

    def forward(self, bridge: HierarchyTokenBridgeInput) -> torch.Tensor:
        batch, count, _ = bridge.block_tokens.shape
        if bridge.block_state.shape != (batch, count, 3):
            raise ValueError("block_state must have shape (batch, n, 3)")
        if bridge.global_prediction.shape != (batch, 4):
            raise ValueError("global_prediction must have shape (batch, 4)")
        if bridge.structural_prediction.shape != (
            batch,
            bridge.structural_tokens.shape[1],
            3,
        ):
            raise ValueError("structural_prediction shape does not match tokens")
        if bridge.structural_mask.shape != (
            batch,
            bridge.structural_tokens.shape[1],
        ):
            raise ValueError("structural_mask shape does not match tokens")
        if bridge.time.shape != (batch,):
            raise ValueError("time must have shape (batch,)")

        query = (
            self.block_norm(bridge.block_tokens)
            + self.time_encoder(bridge.time.float()[:, None])[:, None]
        )
        if not self.bounded:
            query = query + self.state_encoder(bridge.block_state.float())
        structural = bridge.structural_tokens + self.structural_encoder(
            bridge.structural_prediction.float()
        )
        global_token = self.global_encoder(
            bridge.global_prediction.float()
        )[:, None]
        memory = torch.cat([global_token, structural], dim=1)
        allowed = torch.cat(
            [
                torch.ones(
                    batch,
                    count,
                    1,
                    device=query.device,
                    dtype=torch.bool,
                ),
                bridge.structural_mask[:, None].expand(batch, count, -1),
            ],
            dim=-1,
        )
        mixed = self.cross(query, memory, allowed)
        residual = self.update(query + mixed)
        if self.bounded:
            residual = torch.tanh(residual)
        residual = self.gate * residual
        if self.structural_only:
            if bridge.membership is None:
                raise ValueError("structural bridge requires block membership")
            active = (
                bridge.membership[..., :4].any(dim=-1)
                | bridge.membership[..., 5:9].any(dim=-1)
            )
            residual = residual * active[..., None]
        return residual


class TokenBridgeBackbone(SetTransformerDenoiser):
    """Reuse the pinned Trinity backbone with optional per-step hierarchy tokens."""

    def __init__(
        self,
        config: DenoiserArchConfig,
        *,
        bridge_heads: int,
        bridge_mode: str = "legacy",
    ) -> None:
        super().__init__(config)
        if bridge_mode not in {"legacy", "bounded", "structural"}:
            raise ValueError(f"unknown hierarchy bridge mode: {bridge_mode}")
        self.hierarchy_bridge = HierarchyTokenBridge(
            config.hidden,
            bridge_heads,
            bounded=bridge_mode == "bounded",
            structural_only=bridge_mode == "structural",
        )

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: DenoiserCond,
        *,
        bridge: HierarchyTokenBridgeInput | None = None,
    ) -> torch.Tensor:
        x = self.input_embed(z_t, cond.features, cond.graph_pe)
        if bridge is not None:
            x = x + self.hierarchy_bridge(bridge)
        c = self.t_embedder(t)
        bias = self.graph_bias(cond.adjacency, cond.key_pad_mask)
        x, bias = self.time_cond.enter(
            x,
            c,
            bias,
            key_pad_mask=cond.key_pad_mask,
        )
        mods = self.time_cond.modulation(c)
        adj = cond.adjacency
        if adj is not None and self.time_cond_prepends_token:
            adj = nn.functional.pad(adj, (1, 0, 1, 0))
        for block, mod in zip(self.blocks, mods, strict=True):
            x = self._run_block(block, x, mod, bias, adj)
        x = self.time_cond.exit(x)
        x = self.post_norm(x)
        return self.final(x, c)


__all__ = [
    "HierarchyTokenBridge",
    "HierarchyTokenBridgeInput",
    "TokenBridgeBackbone",
]

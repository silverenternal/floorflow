"""Chip adapter for the pinned Trinity block-token denoiser."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

TRINITY_SOURCE = Path(__file__).resolve().parents[3] / "third_party/Trinity/src"
if str(TRINITY_SOURCE) not in sys.path:
    sys.path.insert(0, str(TRINITY_SOURCE))

from trinity.conditioning.graph_pe import SpectralDrawPE
from trinity.models.backbone import DenoiserCond, SetTransformerDenoiser
from trinity.models.presets import DenoiserArchConfig

from floorflow.chip_flow.graph_conditioning import graph_conditioned_features


@dataclass(frozen=True)
class TrinityTokenContext:
    """Prepared graph conditioning consumed by the Trinity denoiser."""

    features: torch.Tensor
    adjacency: torch.Tensor
    key_mask: torch.Tensor
    graph_pe: torch.Tensor


class ChipTrinityTokenFlowModel(nn.Module):
    """Reuse Trinity's token transformer behind the chip flow adapter."""

    def __init__(
        self,
        feature_dim: int = 19,
        hidden: int = 192,
        depth: int = 12,
        heads: int = 3,
        head_dim: int = 64,
        graph_pe_k: int = 2,
    ) -> None:
        super().__init__()
        if feature_dim != 19:
            raise ValueError("Trinity token flow requires the 19-column chip schema")
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.graph_pe_k = int(graph_pe_k)
        self.output_mode = "x0"
        self.backbone = SetTransformerDenoiser(
            DenoiserArchConfig(
                latent_dim=3,
                feature_dim=self.feature_dim,
                graph_pe_dim=self.graph_pe_k,
                hidden=self.hidden,
                depth=self.depth,
                heads=self.heads,
                head_dim=self.head_dim,
                norm="rmsnorm",
                mlp="swiglu",
                mlp_ratio=4.0,
                attn="sdpa_graph",
                qk_norm=True,
                graph_bias=True,
                graph_mix={"name": "mp", "normalize": "row"},
                time_cond="token",
                output_kind="x0",
                post_norm=False,
                time_scale=1000.0,
            )
        )
        # graph_conditioned_features already applies log1p normalization.
        self.graph_pe = SpectralDrawPE(k=self.graph_pe_k, normalize=None)

    def prepare_context(self, batch: dict) -> TrinityTokenContext:
        cached = batch.get("_trinity_token_context")
        if cached is not None and cached[0] is self:
            return cached[1]
        features, adjacency, key_mask = graph_conditioned_features(batch)
        context = TrinityTokenContext(
            features=features,
            adjacency=adjacency,
            key_mask=key_mask,
            graph_pe=self.graph_pe.batched(adjacency, key_mask),
        )
        batch["_trinity_token_context"] = (self, context)
        return context

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        context: TrinityTokenContext,
    ) -> torch.Tensor:
        if state.shape[:2] != context.features.shape[:2]:
            raise ValueError("state and conditioning token sets must match")
        return self.backbone(
            state.float(),
            time.float(),
            DenoiserCond(
                features=context.features,
                adjacency=context.adjacency,
                key_pad_mask=context.key_mask,
                graph_pe=context.graph_pe,
            ),
        )

    def decode(
        self,
        state: torch.Tensor,
        _context: TrinityTokenContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return state, {}


__all__ = ["ChipTrinityTokenFlowModel", "TrinityTokenContext"]

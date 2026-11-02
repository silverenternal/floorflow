"""Learned revision family, operation, and region state."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.model.official_mita_route import OfficialMiTARouteAttention


class RevisionStateHead(nn.Module):
    """Encode an edit request into typed revision state."""

    def __init__(
        self,
        feature_channels: int,
        *,
        hidden_dim: int = 64,
        operations: int = 8,
        families: int = 4,
        token_grid: int = 8,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        self.token_grid = int(token_grid)
        self.feature = nn.Sequential(
            nn.Conv2d(int(feature_channels), hidden, 1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
        )
        self.edit = nn.Conv2d(1, hidden, 1)
        self.operation_queries = nn.Parameter(
            torch.randn(int(operations), hidden) * 0.02
        )
        self.route = OfficialMiTARouteAttention(
            hidden,
            num_heads=4,
            router_count=min(8, int(operations)),
            top_k=min(4, int(operations)),
        )
        self.norm = nn.LayerNorm(hidden)
        self.operation_head = nn.Linear(hidden, int(operations))
        self.family_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(families)),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.region = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.region[-1].weight)
        nn.init.zeros_(self.region[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("features must be [B,C,H,W]")
        if edit_mask.ndim == 3:
            edit_mask = edit_mask[:, None]
        edit_mask = F.interpolate(
            edit_mask.float(),
            size=features.shape[-2:],
            mode="nearest",
        )
        hidden = self.feature(features) + self.edit(edit_mask)
        memory = F.adaptive_avg_pool2d(
            hidden,
            (self.token_grid, self.token_grid),
        ).flatten(2).transpose(1, 2)
        queries = self.operation_queries[None].expand(
            features.shape[0],
            -1,
            -1,
        )
        routed, diagnostics = self.route(queries, memory)
        operation_tokens = self.norm(queries + routed)
        token_logits = self.operation_head(operation_tokens)
        operation_weights = token_logits.softmax(dim=-1).mean(dim=-1)
        operation_logits = token_logits.mean(dim=1)
        operation_context = torch.einsum(
            "bo,bod->bd",
            operation_weights,
            operation_tokens,
        )
        edit_context = (
            hidden * edit_mask
        ).flatten(2).sum(dim=-1) / edit_mask.flatten(1).sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        family_features = torch.cat(
            [operation_context, edit_context],
            dim=-1,
        )
        return {
            "operation_logits": operation_logits,
            "operation_tokens": operation_tokens,
            "family_logits": self.family_head(family_features),
            "revision_gate": torch.sigmoid(
                self.gate(family_features)
            ),
            "region_logits": self.region(hidden),
            "routing_entropy": diagnostics["routing_entropy"],
            "attention_entropy": diagnostics["attention_entropy"],
        }


def revision_state_loss(
    prediction: dict[str, torch.Tensor],
    target_operation: torch.Tensor,
    target_family: torch.Tensor,
    target_region: torch.Tensor,
    *,
    operation_weight: float = 0.5,
    family_weight: float = 1.0,
    region_weight: float = 0.5,
    gate_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    operation = F.cross_entropy(
        prediction["operation_logits"],
        target_operation.long(),
    )
    family = F.cross_entropy(
        prediction["family_logits"],
        target_family.long(),
    )
    region_logits = prediction["region_logits"]
    region_target = F.interpolate(
        target_region.float()[:, None]
        if target_region.ndim == 3
        else target_region.float(),
        size=region_logits.shape[-2:],
        mode="nearest",
    )
    region = F.binary_cross_entropy_with_logits(
        region_logits,
        region_target,
    )
    gate_target = region_target.flatten(1).amax(
        dim=1,
        keepdim=True,
    )
    gate = F.binary_cross_entropy(
        prediction["revision_gate"].clamp(1.0e-6, 1.0 - 1.0e-6),
        gate_target,
    )
    total = (
        float(operation_weight) * operation
        + float(family_weight) * family
        + float(region_weight) * region
        + float(gate_weight) * gate
    )
    return total, {
        "revision_operation_loss": operation.detach(),
        "revision_family_loss": family.detach(),
        "revision_region_loss": region.detach(),
        "revision_gate_loss": gate.detach(),
    }


__all__ = ["RevisionStateHead", "revision_state_loss"]

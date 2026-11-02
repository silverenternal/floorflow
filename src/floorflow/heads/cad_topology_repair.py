"""Typed, bounded graph repair for final CAD wall/opening states."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.heads.attention_masks import unmask_first_empty_row


class CadTopologyRepairHead(nn.Module):
    """Predict wall deletion, endpoint snapping, and opening-host changes."""

    def __init__(
        self,
        candidate_dim: int,
        validation_dim: int,
        *,
        hidden_dim: int = 64,
        heads: int = 4,
        max_endpoint_delta: float = 0.02,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            heads = 1
        self.max_endpoint_delta = float(max_endpoint_delta)
        self.candidate = nn.Linear(int(candidate_dim) + 4, hidden_dim)
        self.geometry = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.validation = nn.Linear(validation_dim, hidden_dim)
        self.diagnostics = nn.Linear(6, hidden_dim)
        self.film = nn.Linear(validation_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            batch_first=True,
        )
        self.keep = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.endpoint_delta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.host_residual_scale = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.keep[-1].weight)
        nn.init.constant_(self.keep[-1].bias, 6.0)
        nn.init.zeros_(self.endpoint_delta[-1].weight)
        nn.init.zeros_(self.endpoint_delta[-1].bias)

    def forward(
        self,
        candidate_features: torch.Tensor,
        endpoints: torch.Tensor,
        valid: torch.Tensor,
        wall_probability: torch.Tensor,
        junction_degree: torch.Tensor,
        opening_attachment: torch.Tensor,
        validity_diagnostics: torch.Tensor,
        validation_token: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return repair logits without applying unbounded graph mutations."""

        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must be [B,N,D]")
        if endpoints.shape[:2] != candidate_features.shape[:2]:
            raise ValueError("candidate features and endpoints disagree")
        if opening_attachment.shape[0] != candidate_features.shape[0] or (
            opening_attachment.shape[-1] != candidate_features.shape[1]
        ):
            raise ValueError("opening attachment must be [B,Q,N]")
        if validity_diagnostics.ndim != 2 or validity_diagnostics.shape[-1] != 6:
            raise ValueError("validity diagnostics must be [B,6]")
        wall_probability = wall_probability.reshape(
            wall_probability.shape[0],
            -1,
        )
        junction_degree = junction_degree.reshape(
            junction_degree.shape[0],
            -1,
        )
        if wall_probability.shape != endpoints.shape[:2] or (
            junction_degree.shape != endpoints.shape[:2]
        ):
            raise ValueError("wall/junction diagnostics must be [B,N]")
        scalar = torch.stack(
            [
                wall_probability.float(),
                junction_degree.float(),
                endpoints[..., 0, 0] - endpoints[..., 1, 0],
                endpoints[..., 0, 1] - endpoints[..., 1, 1],
            ],
            dim=-1,
        )
        tokens = (
            self.candidate(
                torch.cat(
                    [
                        candidate_features.float(),
                        scalar,
                    ],
                    dim=-1,
                )
            )
            + self.geometry(endpoints.flatten(2).float())
            + self.validation(validation_token.float())[:, None]
            + self.diagnostics(validity_diagnostics.float())[:, None]
        )
        scale, bias = self.film(validation_token.float()).chunk(2, dim=-1)
        tokens = tokens * (1.0 + scale.tanh())[:, None] + bias.tanh()[:, None]
        valid_mask = valid.bool()
        attention_mask = ~valid_mask
        attention_mask = unmask_first_empty_row(attention_mask)
        memory = self.norm(tokens)
        attended = self.attention(
            memory,
            memory,
            memory,
            key_padding_mask=attention_mask,
            need_weights=False,
        )[0]
        tokens = tokens + attended
        keep_logits = self.keep(tokens)[..., 0]
        endpoint_delta = (
            self.endpoint_delta(tokens).reshape(
                *endpoints.shape[:2],
                2,
                2,
            )
            .tanh()
            * self.max_endpoint_delta
        )
        host_probability = opening_attachment.float().clamp_min(1.0e-6)
        host_probability = host_probability / host_probability.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        opening_context = torch.einsum(
            "bqn,bnd->bqd",
            host_probability,
            tokens,
        )
        host_residual = torch.einsum(
            "bqd,bnd->bqn",
            opening_context,
            tokens,
        ) / max(1, tokens.shape[-1]) ** 0.5
        host_logits = host_probability.clamp_min(1.0e-6).log()
        host_logits = (
            host_logits
            + self.host_residual_scale.tanh() * host_residual
        )
        return {
            "wall_keep_logits": keep_logits,
            "endpoint_delta": endpoint_delta.masked_fill(
                ~valid_mask[..., None, None],
                0.0,
            ),
            "opening_host_logits": host_logits,
        }


@torch.no_grad()
def apply_bounded_topology_repair(
    endpoints: torch.Tensor,
    valid: torch.Tensor,
    keep_logits: torch.Tensor,
    endpoint_delta: torch.Tensor,
    *,
    keep_threshold: float = 0.5,
    move_threshold: float = 1.0e-4,
    max_edits: int = 8,
) -> dict[str, torch.Tensor]:
    """Apply only deterministic typed edits with a hard per-sample budget."""

    if endpoints.ndim != 4 or endpoints.shape[2:] != (2, 2):
        raise ValueError("endpoints must be [B,N,2,2]")
    if endpoint_delta.shape != endpoints.shape:
        raise ValueError("endpoint delta must match endpoints")
    valid_mask = valid.bool()
    remove_risk = (1.0 - keep_logits.sigmoid()).masked_fill(
        ~valid_mask,
        -1.0,
    )
    remove = valid_mask & (keep_logits.sigmoid() < float(keep_threshold))
    move_risk = endpoint_delta.float().square().sum(dim=(2, 3)).sqrt()
    move = valid_mask & (move_risk > float(move_threshold))
    risk = torch.where(remove, remove_risk, torch.zeros_like(remove_risk))
    risk = torch.where(
        move & ~remove,
        move_risk,
        risk,
    )
    budget = max(0, int(max_edits))
    if budget < risk.shape[1]:
        allowed = torch.zeros_like(risk, dtype=torch.bool)
        order = risk.argsort(dim=1, descending=True)[:, :budget]
        allowed.scatter_(1, order, True)
        over_budget = ~allowed
        remove = remove & ~over_budget
        move = move & ~over_budget
    repaired = endpoints + endpoint_delta.float() * move[..., None, None]
    edited = remove | move
    return {
        "endpoints": repaired,
        "valid": valid_mask & ~remove,
        "edited": edited,
        "removed": remove,
        "moved": move,
    }


__all__ = ["CadTopologyRepairHead", "apply_bounded_topology_repair"]

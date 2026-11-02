"""Learn which room-boundary wall candidates belong to the CAD wall layer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.objectives.candidate_filtration_loss import endpoint_adjacency


def _coordinate_features(points: torch.Tensor) -> torch.Tensor:
    angles = points * (2.0 * math.pi)
    return torch.cat(
        [
            points,
            torch.sin(angles),
            torch.cos(angles),
        ],
        dim=-1,
    )


class CadWallCandidateScorer(nn.Module):
    """Score and locally refine fixed room-boundary wall proposals."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 96,
        graph_dim: int = 96,
        residual_scale: float = 0.01,
        descriptor_dim: int = 8,
        attention_heads: int = 4,
        use_encoded_score: bool = False,
        source_count: int = 6,
        source_features: bool = False,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
            nn.SiLU(),
        )
        self.source_embedding = (
            nn.Embedding(int(source_count), hidden_dim)
            if source_features
            else None
        )
        if self.source_embedding is not None:
            nn.init.normal_(self.source_embedding.weight, std=0.02)
        self.graph_projection = nn.Linear(graph_dim, hidden_dim)
        self.fallback_descriptor_projection = nn.Linear(hidden_dim, graph_dim)
        self.descriptor_predictor = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, descriptor_dim),
        )
        self.descriptor_film = nn.Linear(descriptor_dim + 3, hidden_dim * 2)
        self.phase_embedding = nn.Embedding(2, hidden_dim)
        self.graph_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            batch_first=True,
        )
        geometry_dim = 21
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3 + geometry_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.use_encoded_score = bool(use_encoded_score)
        score_input_dim = 12 + (hidden_dim if self.use_encoded_score else 0)
        self.score = nn.Sequential(
            nn.Linear(score_input_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.evidence_scale = nn.Parameter(torch.zeros(1))
        self.wall_type = nn.Linear(hidden_dim, 1)
        self.budget = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.phase_budget = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for name in ("exterior", "interior")
            }
        )
        self.residual = nn.Linear(hidden_dim, 4)
        self.residual_scale = float(residual_scale)
        nn.init.xavier_uniform_(self.score[0].weight)
        nn.init.zeros_(self.score[0].bias)
        nn.init.zeros_(self.score[2].weight)
        nn.init.zeros_(self.score[2].bias)
        nn.init.zeros_(self.wall_type.weight)
        nn.init.zeros_(self.wall_type.bias)
        nn.init.xavier_uniform_(self.budget[0].weight)
        nn.init.zeros_(self.budget[0].bias)
        nn.init.normal_(self.budget[2].weight, std=0.01)
        nn.init.constant_(self.budget[2].bias, -1.4)
        for head in self.phase_budget.values():
            nn.init.xavier_uniform_(head[0].weight)
            nn.init.zeros_(head[0].bias)
            nn.init.normal_(head[2].weight, std=0.01)
            nn.init.constant_(head[2].bias, -2.0)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def _sample(
        self,
        features: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        grid = points * 2.0 - 1.0
        sampled = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.permute(0, 2, 3, 1)

    def forward(
        self,
        features: torch.Tensor,
        endpoints: torch.Tensor,
        valid: torch.Tensor,
        evidence: torch.Tensor,
        graph_tokens: torch.Tensor | None = None,
        exterior: torch.Tensor | None = None,
        proposal_source: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if endpoints.ndim != 4 or endpoints.shape[-2:] != (2, 2):
            raise ValueError("endpoints must be [B,N,2,2]")
        hidden = self.feature_projection(features)
        midpoint = 0.5 * (endpoints[:, :, 0] + endpoints[:, :, 1])
        samples = torch.stack(
            [endpoints[:, :, 0], midpoint, endpoints[:, :, 1]],
            dim=2,
        )
        sampled = self._sample(hidden, samples).flatten(start_dim=2)
        delta = endpoints[:, :, 1] - endpoints[:, :, 0]
        length = delta.norm(dim=-1, keepdim=True)
        direction = delta / length.clamp_min(1.0e-6)
        geometry = torch.cat(
            [
                endpoints.flatten(start_dim=2),
                _coordinate_features(endpoints[:, :, 0]),
                _coordinate_features(endpoints[:, :, 1]),
                length,
                direction,
                evidence[..., None],
                valid[..., None].to(endpoints.dtype),
            ],
            dim=-1,
        )
        encoded = self.encoder(torch.cat([sampled, geometry], dim=-1))
        if proposal_source is not None and self.source_embedding is not None:
            if proposal_source.shape != valid.shape:
                raise ValueError("proposal_source must match [B,N] candidates")
            encoded = encoded + self.source_embedding(
                proposal_source.long().clamp(0, self.source_embedding.num_embeddings - 1)
            )
        phase = (
            torch.zeros_like(valid, dtype=torch.long)
            if exterior is None
            else exterior.long()
        )
        encoded = encoded + self.phase_embedding(phase)
        adjacency = endpoint_adjacency(endpoints, valid)
        attention_bias = torch.where(
            adjacency > 0.5,
            torch.zeros_like(adjacency),
            torch.full_like(adjacency, -4.0),
        )
        attention_bias = attention_bias[:, None].expand(
            endpoints.shape[0],
            self.graph_attention.num_heads,
            endpoints.shape[1],
            endpoints.shape[1],
        ).reshape(
            endpoints.shape[0] * self.graph_attention.num_heads,
            endpoints.shape[1],
            endpoints.shape[1],
        )
        padding = torch.where(
            ~valid,
            torch.full_like(encoded[..., 0], float("-inf")),
            torch.zeros_like(encoded[..., 0]),
        )
        empty = ~valid.all(dim=1)
        if bool(empty.any()):
            padding = padding.clone()
            padding[empty, 0] = 0.0
        graph_context = self.graph_attention(
            encoded,
            encoded,
            encoded,
            attn_mask=attention_bias,
            key_padding_mask=padding,
            need_weights=False,
        )[0]
        encoded = encoded + graph_context * valid[..., None].to(encoded.dtype)
        if graph_tokens is not None:
            token = self.graph_projection(graph_tokens.mean(dim=1))
            encoded = encoded + token[:, None]
            descriptor_source = graph_tokens.mean(dim=1)
        else:
            descriptor_source = self.fallback_descriptor_projection(
                encoded.mean(dim=1)
            )
        valid_float = valid[..., None].to(encoded.dtype)
        global_feature = (encoded * valid_float).sum(dim=1) / (
            valid_float.sum(dim=1).clamp_min(1.0)
        )
        budget_fraction = self.budget(global_feature).squeeze(-1).sigmoid()
        phase_fractions = {
            name: head(global_feature).squeeze(-1).sigmoid()
            for name, head in self.phase_budget.items()
        }
        predicted_descriptor = self.descriptor_predictor(descriptor_source)
        descriptor_control = torch.cat(
            [
                predicted_descriptor / float(endpoints.shape[1]),
                budget_fraction[..., None],
                phase_fractions["exterior"][..., None],
                phase_fractions["interior"][..., None],
            ],
            dim=-1,
        )
        scale, bias = self.descriptor_film(descriptor_control).chunk(2, dim=-1)
        encoded = (
            encoded * (1.0 + 0.25 * torch.tanh(scale[:, None]))
            + 0.25 * torch.tanh(bias[:, None])
        )
        residual = (
            torch.tanh(self.residual(encoded)).reshape_as(endpoints)
            * valid[..., None, None].to(endpoints.dtype)
            * self.residual_scale
        )
        evidence_logits = torch.logit(
            evidence.clamp(1.0e-4, 1.0 - 1.0e-4)
        )
        degree = adjacency.sum(dim=-1) / 4.0
        score_features = torch.cat(
            [
                evidence_logits[..., None],
                torch.log1p(length),
                phase.to(encoded.dtype)[..., None],
                degree[..., None],
                (
                    predicted_descriptor
                    / float(endpoints.shape[1])
                )[:, None].expand(-1, endpoints.shape[1], -1),
            ],
            dim=-1,
        )
        if self.use_encoded_score:
            score_features = torch.cat(
                [score_features, encoded],
                dim=-1,
            )
        raw_score = self.score(score_features).squeeze(-1)
        evidence_scale = 1.0 + 0.5 * torch.tanh(self.evidence_scale)
        score_residual = torch.tanh(raw_score)
        score_logits = score_residual + evidence_scale * evidence_logits
        return {
            "score_logits": score_logits,
            "raw_score_logits": raw_score,
            "score_residual_logits": score_residual,
            "evidence_scale": evidence_scale.expand(endpoints.shape[0]),
            "candidate_tokens": encoded,
            "exterior_logits": self.wall_type(encoded).squeeze(-1),
            "budget_fraction": budget_fraction,
            "budget_count": budget_fraction * float(endpoints.shape[1]),
            "predicted_descriptor": predicted_descriptor,
            "exterior_budget_fraction": phase_fractions["exterior"],
            "exterior_budget_count": (
                phase_fractions["exterior"] * float(endpoints.shape[1])
            ),
            "interior_budget_fraction": phase_fractions["interior"],
            "interior_budget_count": (
                phase_fractions["interior"] * float(endpoints.shape[1])
            ),
            "endpoints": (endpoints + residual).clamp(0.0, 1.0),
            "residual": residual,
        }


__all__ = ["CadWallCandidateScorer"]

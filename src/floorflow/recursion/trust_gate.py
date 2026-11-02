"""Train-only gate for trusting recursive graph geometry."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

TRUST_FEATURE_NAMES = (
    "base_confidence",
    "graph_confidence",
    "confidence_gain",
    "support_agreement",
    "base_entropy",
    "graph_entropy",
    "base_coverage",
    "graph_coverage",
)


class RoomGraphTrustGate(nn.Module):
    """Predict whether graph geometry is likely better than the base branch."""

    def __init__(
        self,
        feature_size: int = len(TRUST_FEATURE_NAMES),
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.feature_size = int(feature_size)
        self.network = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_size:
            raise ValueError("trust features must be [B,F]")
        return self.network(features.float()).squeeze(-1)

    @torch.no_grad()
    def trust(self, features: torch.Tensor) -> torch.Tensor:
        return self(features).sigmoid()


def load_room_graph_trust_gate(
    path: str | Path,
    device: torch.device,
) -> RoomGraphTrustGate:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    gate = RoomGraphTrustGate(
        feature_size=int(
            config.get("feature_size", len(TRUST_FEATURE_NAMES))
        ),
        hidden_size=int(config.get("hidden_size", 32)),
    )
    gate.load_state_dict(payload["model"])
    return gate.to(device).eval()


def room_logit_trust_features(
    base_logits: torch.Tensor,
    graph_logits: torch.Tensor,
    footprint: torch.Tensor,
) -> torch.Tensor:
    """Summarize target-free evidence for graph trust per sample."""

    if base_logits.shape != graph_logits.shape:
        raise ValueError("base and graph room logits must match")
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    base_probability = base_logits.float().softmax(dim=1)
    graph_probability = graph_logits.float().softmax(dim=1)
    valid = footprint.bool()
    denominator = valid.flatten(1).sum(dim=1).clamp_min(1.0)
    base_confidence = base_probability.amax(dim=1)
    graph_confidence = graph_probability.amax(dim=1)
    base_entropy = -(
        base_probability.clamp_min(1.0e-8)
        * base_probability.clamp_min(1.0e-8).log()
    ).sum(dim=1) / math.log(max(2, base_logits.shape[1]))
    graph_entropy = -(
        graph_probability.clamp_min(1.0e-8)
        * graph_probability.clamp_min(1.0e-8).log()
    ).sum(dim=1) / math.log(max(2, graph_logits.shape[1]))
    base_labels = base_probability.argmax(dim=1)
    graph_labels = graph_probability.argmax(dim=1)
    agreement = (base_labels == graph_labels) & valid
    base_coverage = (base_confidence >= 0.5) & valid
    graph_coverage = (graph_confidence >= 0.5) & valid

    def masked_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * valid).flatten(1).sum(dim=1) / denominator

    return torch.stack(
        [
            masked_mean(base_confidence),
            masked_mean(graph_confidence),
            masked_mean(graph_confidence - base_confidence),
            agreement.flatten(1).sum(dim=1) / denominator,
            masked_mean(base_entropy),
            masked_mean(graph_entropy),
            base_coverage.flatten(1).sum(dim=1) / denominator,
            graph_coverage.flatten(1).sum(dim=1) / denominator,
        ],
        dim=-1,
    )


__all__ = [
    "TRUST_FEATURE_NAMES",
    "RoomGraphTrustGate",
    "load_room_graph_trust_gate",
    "room_logit_trust_features",
]

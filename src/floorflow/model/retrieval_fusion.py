"""Learned, zero-initialized fusion of model and contour-retrieved room logits."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    return -(
        probability.clamp_min(1.0e-8)
        * probability.clamp_min(1.0e-8).log()
    ).sum(dim=1, keepdim=True) / max(
        1.0e-6,
        float(probability.shape[1]),
    )


class LearnedRetrievalFusion(nn.Module):
    """Predict a per-class spatial gate for a train-only retrieval prior."""

    def __init__(
        self,
        num_room_classes: int,
        hidden_channels: int = 32,
        gate_resolution: int = 64,
        base_prior_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_room_classes = int(num_room_classes)
        self.gate_resolution = int(gate_resolution)
        self.base_prior_weight = nn.Parameter(
            torch.tensor(float(base_prior_weight))
        )
        feature_channels = self.num_room_classes * 2 + 6
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(max(1, hidden_channels // 8), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, self.num_room_classes, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        model_logits: torch.Tensor,
        retrieval_logits: torch.Tensor,
        footprint: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if model_logits.shape != retrieval_logits.shape:
            raise ValueError("model and retrieval logits must have equal shape")
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        model_probability = model_logits.softmax(dim=1)
        retrieval_probability = retrieval_logits.sigmoid()
        model_top = model_probability.topk(2, dim=1).values
        retrieval_top = retrieval_probability.topk(2, dim=1).values
        agreement = (
            model_probability.argmax(dim=1, keepdim=True)
            == retrieval_probability.argmax(dim=1, keepdim=True)
        ).float()
        features = torch.cat(
            [
                model_probability,
                retrieval_probability,
                _entropy(model_probability),
                _entropy(retrieval_probability),
                (model_top[:, :1] - model_top[:, 1:2]),
                (retrieval_top[:, :1] - retrieval_top[:, 1:2]),
                agreement,
                footprint.float(),
            ],
            dim=1,
        )
        gate = self.gate(
            F.adaptive_avg_pool2d(
                features,
                (self.gate_resolution, self.gate_resolution),
            )
        )
        gate = F.interpolate(
            gate,
            size=model_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = (
            model_logits
            + self.base_prior_weight.to(model_logits.dtype) * retrieval_logits
            + gate * retrieval_logits
        )
        return fused, {
            "gate": gate,
            "gate_abs_mean": gate.abs().mean(),
            "gate_active_fraction": (gate.abs() > 0.01).float().mean(),
            "retrieval_agreement": agreement.mean(),
            "base_prior_weight": self.base_prior_weight,
        }


__all__ = ["LearnedRetrievalFusion"]

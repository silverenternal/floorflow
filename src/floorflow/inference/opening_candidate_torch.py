"""GPU candidate verifier for opening interval proposals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class OpeningCandidateMLP(nn.Module):
    def __init__(self, feature_count: int, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(int(feature_count), int(hidden)),
            nn.SiLU(),
            nn.LayerNorm(int(hidden)),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
        )
        self.presence = nn.Linear(int(hidden), 1)
        self.family = nn.Linear(int(hidden), 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.presence(hidden).squeeze(-1), self.family(hidden)


@dataclass(frozen=True)
class OpeningCandidateTorchRanker:
    model: OpeningCandidateMLP
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    presence_threshold: float
    family_mode: str = "model"

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def probabilities(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if features.size == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )
        value = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        value = (value - self.feature_mean) / self.feature_std
        presence_logits, family_logits = self.model(value)
        return (
            presence_logits.sigmoid().cpu().numpy(),
            family_logits.softmax(dim=-1).cpu().numpy(),
        )


def save_opening_candidate_torch_ranker(
    path: str | Path,
    ranker: OpeningCandidateTorchRanker,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": ranker.model.state_dict(),
        "feature_count": next(ranker.model.parameters()).shape[-1],
        "hidden": ranker.model.trunk[0].out_features,
        "feature_mean": ranker.feature_mean.cpu(),
        "feature_std": ranker.feature_std.cpu(),
        "presence_threshold": float(ranker.presence_threshold),
        "family_mode": ranker.family_mode,
    }
    torch.save(payload, destination)


def load_opening_candidate_torch_ranker(
    path: str | Path,
    device: str | torch.device,
) -> OpeningCandidateTorchRanker:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = OpeningCandidateMLP(
        int(payload["feature_count"]),
        int(payload["hidden"]),
    )
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    return OpeningCandidateTorchRanker(
        model=model,
        feature_mean=payload["feature_mean"].to(device),
        feature_std=payload["feature_std"].to(device),
        presence_threshold=float(payload["presence_threshold"]),
        family_mode=str(payload.get("family_mode", "model")),
    )


def train_opening_candidate_mlp(
    features: np.ndarray,
    presence: np.ndarray,
    family: np.ndarray,
    *,
    device: str | torch.device,
    hidden: int = 64,
    epochs: int = 200,
    learning_rate: float = 1e-3,
) -> OpeningCandidateTorchRanker:
    x = np.asarray(features, dtype=np.float32)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1.0e-5)
    tensor = torch.as_tensor((x - mean) / std, dtype=torch.float32, device=device)
    presence_target = torch.as_tensor(presence, dtype=torch.float32, device=device)
    family_target = torch.as_tensor(family, dtype=torch.long, device=device)
    positive = family_target >= 0
    model = OpeningCandidateMLP(x.shape[1], hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    presence_weight = torch.as_tensor(
        [len(presence) / max(1.0, 2.0 * presence.sum()),
         len(presence) / max(1.0, 2.0 * (len(presence) - presence.sum()))],
        dtype=torch.float32,
        device=device,
    )
    family_values = family_target[positive]
    family_weight = torch.as_tensor(
        [
            len(family_values) / max(1.0, 2.0 * (family_values == 0).sum()),
            len(family_values) / max(1.0, 2.0 * (family_values == 1).sum()),
        ],
        dtype=torch.float32,
        device=device,
    )
    for _epoch in range(max(1, int(epochs))):
        model.train()
        predicted_presence, predicted_family = model(tensor)
        presence_loss = F.binary_cross_entropy_with_logits(
            predicted_presence,
            presence_target,
            weight=presence_weight[presence_target.long()],
        )
        family_loss = F.cross_entropy(
            predicted_family[positive],
            family_target[positive],
            weight=family_weight,
        )
        loss = presence_loss + family_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return OpeningCandidateTorchRanker(
        model=model,
        feature_mean=torch.as_tensor(mean, dtype=torch.float32, device=device),
        feature_std=torch.as_tensor(std, dtype=torch.float32, device=device),
        presence_threshold=0.5,
    )


__all__ = [
    "OpeningCandidateMLP",
    "OpeningCandidateTorchRanker",
    "load_opening_candidate_torch_ranker",
    "save_opening_candidate_torch_ranker",
    "train_opening_candidate_mlp",
]

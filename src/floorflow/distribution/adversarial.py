"""DMD-style adversarial adaptation for generated floorplan distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _gumbel_softmax_topk(
    logits: torch.Tensor,
    ratio: float,
    *,
    training: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("mask ratio must be in (0, 1]")
    noise = torch.rand_like(logits, generator=generator) if training else 0.0
    perturbed = logits + noise
    count = max(1, math.ceil(logits.shape[1] * ratio))
    hard = torch.zeros_like(perturbed)
    indices = perturbed.topk(count, dim=1).indices
    hard.scatter_(1, indices, 1.0)
    soft = torch.sigmoid(logits)
    return hard + soft - soft.detach()


class DynamicFeatureMaskDiscriminator(nn.Module):
    """Discriminator that masks drifting features instead of forgetting them."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 32,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        self.channels = int(channels)
        self.mask_ratio = float(mask_ratio)
        self.input_projection = nn.Sequential(
            nn.Conv2d(self.channels + 1, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 8), hidden),
            nn.SiLU(),
        )
        self.feature = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 8), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 8), hidden),
            nn.SiLU(),
        )
        self.mask_router = nn.Conv2d(hidden, hidden, 1)
        self.drift_projection = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("feature_ema", torch.zeros(hidden))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))
        self.last_mask_fraction: torch.Tensor | None = None
        self.last_drift: torch.Tensor | None = None

    def forward(
        self,
        distribution: torch.Tensor,
        footprint: torch.Tensor,
        *,
        update_ema: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        inputs = torch.cat(
            [distribution.float(), footprint.float()],
            dim=1,
        )
        projected = self.input_projection(inputs)
        features = self.feature(projected)
        hidden_mean = features.detach().mean(dim=(0, 2, 3))
        if int(self.updates) == 0:
            drift = torch.zeros_like(features[:, :1])
        else:
            drift = (
                features.detach()
                - self.feature_ema[None, :, None, None]
            ).abs().mean(dim=1, keepdim=True)
        normalizer = drift.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-6)
        drift = (drift / normalizer).clamp(0.0, 4.0)
        drift = drift.expand(-1, features.shape[1], -1, -1)
        router_logits = (
            self.mask_router(features)
            + self.drift_projection(drift).to(features.dtype)
        )
        mask = _gumbel_softmax_topk(
            router_logits,
            self.mask_ratio,
            training=self.training,
            generator=generator,
        )
        masked = features * mask
        logits = self.head(masked).squeeze(-1)
        if update_ema:
            decay = 0.99
            self.feature_ema.mul_(decay).add_(hidden_mean, alpha=1.0 - decay)
            self.updates.add_(1)
        self.last_mask_fraction = mask.detach().mean()
        self.last_drift = drift.detach().mean()
        return logits, {
            "mask_fraction": self.last_mask_fraction,
            "feature_drift": self.last_drift,
            "ema_updates": self.updates.detach(),
        }


def hinge_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = F.relu(1.0 - real_logits).mean() + F.relu(
        1.0 + fake_logits
    ).mean()
    return loss, {
        "real_score": real_logits.detach().mean(),
        "fake_score": fake_logits.detach().mean(),
        "real_accuracy": (real_logits.detach() > 0).float().mean(),
        "fake_accuracy": (fake_logits.detach() < 0).float().mean(),
    }


def hinge_generator_loss(
    fake_logits: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -fake_logits.mean()
    return loss, {"generator_adversarial_score": fake_logits.detach().mean()}


@dataclass
class DistributionAdversarialTrainer:
    discriminator: DynamicFeatureMaskDiscriminator
    lr: float = 1.0e-4
    generator_weight: float = 0.02
    r1_weight: float = 0.0

    def __post_init__(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=float(self.lr),
            betas=(0.5, 0.9),
        )

    def generator_loss(
        self,
        fake_distribution: torch.Tensor,
        footprint: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fake_logits, diagnostics = self.discriminator(
            fake_distribution,
            footprint,
            generator=generator,
        )
        loss, metrics = hinge_generator_loss(fake_logits)
        details = dict(metrics)
        details.update(
            {
                "generator_" + key: value
                for key, value in diagnostics.items()
            }
        )
        return float(self.generator_weight) * loss, details

    def discriminator_step(
        self,
        real_distribution: torch.Tensor,
        fake_distribution: torch.Tensor,
        footprint: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        self.optimizer.zero_grad(set_to_none=True)
        real = real_distribution.detach().requires_grad_(
            float(self.r1_weight) > 0.0
        )
        fake = fake_distribution.detach()
        real_logits, real_diag = self.discriminator(
            real,
            footprint,
            update_ema=True,
            generator=generator,
        )
        fake_logits, fake_diag = self.discriminator(
            fake,
            footprint,
            generator=generator,
        )
        loss, metrics = hinge_discriminator_loss(real_logits, fake_logits)
        if float(self.r1_weight) > 0.0:
            gradient = torch.autograd.grad(
                real_logits.sum(),
                real,
                create_graph=True,
            )[0]
            r1 = gradient.float().square().flatten(1).sum(dim=1).mean()
            loss = loss + 0.5 * float(self.r1_weight) * r1
        else:
            r1 = loss.detach() * 0.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.optimizer.step()
        return {
            **metrics,
            "discriminator_loss": loss.detach(),
            "r1_penalty": r1.detach(),
            "real_mask_fraction": real_diag["mask_fraction"],
            "fake_mask_fraction": fake_diag["mask_fraction"],
            "feature_drift": real_diag["feature_drift"],
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr": self.lr,
            "generator_weight": self.generator_weight,
            "r1_weight": self.r1_weight,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])


__all__ = [
    "DistributionAdversarialTrainer",
    "DynamicFeatureMaskDiscriminator",
    "hinge_discriminator_loss",
    "hinge_generator_loss",
]

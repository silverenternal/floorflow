"""Auxiliary legalizer-response head built from existing scale_pack telemetry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _zero_linear(in_features: int, out_features: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class LegalizerResponseHead(nn.Module):
    """Predict per-block displacement and a scale_pack rung distribution."""

    def __init__(self, hidden: int, *, bottleneck: int | None = None) -> None:
        super().__init__()
        width = int(bottleneck or max(16, hidden // 8))
        self.token_encoder = nn.Linear(hidden, width)
        self.state_encoder = nn.Linear(3, width)
        self.prediction_encoder = nn.Linear(3, width)
        self.time_encoder = nn.Linear(hidden, width)
        self.shared = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.SiLU(),
        )
        self.displacement = _zero_linear(width, 2)
        self.rung = _zero_linear(width, 5)

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        block_prediction: torch.Tensor,
        time_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, _ = block_tokens.shape
        context = torch.cat(
            [
                self.token_encoder(block_tokens.float()),
                self.state_encoder(block_state.float()),
                self.prediction_encoder(block_prediction.float()),
                self.time_encoder(time_token.float())[:, None].expand(
                    batch,
                    count,
                    -1,
                ),
            ],
            dim=-1,
        )
        shared = self.shared(context)
        displacement = self.displacement(shared)
        rung_logits = self.rung(shared).mean(dim=1)
        return displacement, rung_logits


@dataclass(frozen=True)
class LegalizerResponseTerms:
    total: torch.Tensor
    displacement: torch.Tensor
    rung: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "legalizer_response": self.total,
            "legalizer_response_displacement": self.displacement,
            "legalizer_response_rung": self.rung,
        }


def legalizer_response_loss(
    predicted_displacement: torch.Tensor,
    predicted_rung_logits: torch.Tensor,
    batch: dict,
    *,
    displacement_weight: float = 1.0,
    rung_weight: float = 0.1,
    huber_beta: float = 0.1,
) -> LegalizerResponseTerms:
    """Supervise the auxiliary head from scale_pack displacement and rung labels."""

    if "legalizer_displacement" not in batch or "legalizer_rung" not in batch:
        raise ValueError("legalizer-response supervision labels are missing")
    target_displacement = batch["legalizer_displacement"].float()
    target_rung = batch["legalizer_rung"].long()
    if predicted_displacement.shape != target_displacement.shape:
        raise ValueError("legalizer displacement target shape mismatch")
    if predicted_rung_logits.shape != (predicted_displacement.shape[0], 5):
        raise ValueError("legalizer rung logits must have five classes")
    displacement_error = F.smooth_l1_loss(
        predicted_displacement.float(),
        target_displacement,
        beta=huber_beta,
        reduction="none",
    )
    mask = batch.get("legalizer_mask")
    if mask is None:
        mask = torch.ones_like(predicted_displacement[..., 0], dtype=torch.bool)
    else:
        mask = torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=predicted_displacement.device,
        )
        if mask.shape != predicted_displacement.shape[:2]:
            raise ValueError("legalizer mask must match block axes")
    constraints = batch.get("constraints")
    if constraints is not None:
        preplaced = constraints[..., 1].ne(0)
        if preplaced.shape != mask.shape:
            raise ValueError("constraints must match legalizer block axes")
        mask = mask & ~preplaced
    expanded = mask[..., None].to(displacement_error.dtype)
    displacement = (
        displacement_error * expanded
    ).sum() / expanded.sum().clamp_min(1.0)
    rung = F.cross_entropy(predicted_rung_logits.float(), target_rung)
    total = (
        float(displacement_weight) * displacement
        + float(rung_weight) * rung
    )
    return LegalizerResponseTerms(
        total=total,
        displacement=displacement,
        rung=rung,
    )


def legalizer_response_targets(
    before_boxes: torch.Tensor,
    after_boxes: torch.Tensor,
    rung: torch.Tensor,
    areas: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Normalize existing boxes and scale_pack rung telemetry into head labels."""

    if before_boxes.shape != after_boxes.shape:
        raise ValueError("before and after legalization boxes must match")
    scale = areas.float().sum(dim=1).sqrt().clamp_min(1.0e-6)
    before_center = before_boxes[..., :2] + before_boxes[..., 2:] / 2.0
    after_center = after_boxes[..., :2] + after_boxes[..., 2:] / 2.0
    displacement = (after_center - before_center) / scale[:, None, None]
    return {
        "legalizer_displacement": displacement,
        "legalizer_rung": rung.long().clamp(0, 4),
    }


def legalizer_response_from_scale_pack(
    before_boxes: torch.Tensor,
    legalized_xywh: torch.Tensor,
    areas: torch.Tensor,
    info: dict,
) -> dict[str, torch.Tensor]:
    """Convert existing ``scale_pack_legalize`` output and ``last_info`` telemetry."""

    rung = info.get("rung")
    if rung is None:
        raise ValueError("scale_pack telemetry is missing rung")
    rung_tensor = torch.as_tensor(rung, device=before_boxes.device).reshape(-1)
    if rung_tensor.numel() == 1:
        rung_tensor = rung_tensor.expand(before_boxes.shape[0])
    return legalizer_response_targets(
        before_boxes,
        legalized_xywh,
        rung_tensor,
        areas,
    )


def apply_legalizer_displacement(
    block_prediction: torch.Tensor,
    displacement: torch.Tensor | None,
    *,
    strength: float = 1.0,
    max_displacement: float = 0.1,
    center_mask: torch.Tensor | None = None,
    per_block_radius: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a per-block legalizer displacement inside a trust radius."""

    if (
        not math.isfinite(float(strength))
        or not math.isfinite(float(max_displacement))
        or not 0.0 <= strength <= 1.0
        or max_displacement < 0.0
    ):
        raise ValueError("legalizer strength/radius is outside its trust region")
    if displacement is None:
        return block_prediction
    if block_prediction.shape[:2] != displacement.shape[:2]:
        raise ValueError("legalizer displacement must match block axes")
    if displacement.shape[-1] != 2 or block_prediction.shape[-1] != 3:
        raise ValueError("legalizer displacement must have two center coordinates")
    if center_mask is not None:
        mask = torch.as_tensor(
            center_mask,
            dtype=torch.bool,
            device=displacement.device,
        )
        if mask.shape != displacement.shape[:2]:
            raise ValueError("legalizer center mask must match block axes")
        displacement = displacement * mask[..., None].to(displacement.dtype)
    radius = torch.full(
        displacement.shape[:2],
        float(max_displacement),
        dtype=displacement.dtype,
        device=displacement.device,
    )
    if per_block_radius is not None:
        supplied = torch.as_tensor(
            per_block_radius,
            dtype=displacement.dtype,
            device=displacement.device,
        )
        if supplied.shape != displacement.shape[:2]:
            raise ValueError("legalizer per-block radius must match block axes")
        radius = torch.minimum(radius, supplied.clamp_min(0.0))
    padded = torch.cat(
        [
            displacement,
            displacement.new_zeros((*displacement.shape[:2], 1)),
        ],
        dim=-1,
    )
    norm = padded.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    trust = (radius[..., None] / norm).clamp_max(1.0)
    return block_prediction + float(strength) * padded * trust


__all__ = [
    "LegalizerResponseHead",
    "LegalizerResponseTerms",
    "apply_legalizer_displacement",
    "legalizer_response_from_scale_pack",
    "legalizer_response_loss",
    "legalizer_response_targets",
]

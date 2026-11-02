"""Full per-pixel class distribution head and IoU-aligned objectives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.algorithmic_dynamic_head import (
    DynamicKernelOutputHead,
    DynamicQueryMaskHead,
)
from floorflow.heads.dynamic_output import DynamicMultiScaleOutputHead


class PixelDistributionHead(nn.Module):
    """Predict a categorical distribution over all CAD classes per pixel."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 96,
        room_ids: tuple[int, ...] | None = None,
        dynamic_head_variant: str = "generated",
    ) -> None:
        super().__init__()
        hidden = max(3, (int(hidden_channels) // 3) * 3)
        self.num_classes = int(num_classes)
        self.room_ids = torch.tensor(room_ids or (), dtype=torch.long)
        self.multi_scale = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        int(in_channels),
                        hidden // 3,
                        kernel_size,
                        padding=kernel_size // 2,
                        groups=math.gcd(int(in_channels), hidden // 3),
                    ),
                    nn.GroupNorm(max(1, (hidden // 3) // 8), hidden // 3),
                    nn.SiLU(),
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 8), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, self.num_classes, 1),
        )
        if dynamic_head_variant == "generated":
            self.dynamic_output = DynamicKernelOutputHead(
                hidden,
                self.num_classes,
                ((1, 1), (3, 1), (5, 1), (7, 1), (3, 2), (5, 2)),
            )
        elif dynamic_head_variant == "mixture":
            self.dynamic_output = DynamicMultiScaleOutputHead(
                hidden,
                self.num_classes,
                (1, 3, 5, 7),
            )
        elif dynamic_head_variant == "query":
            self.dynamic_output = DynamicQueryMaskHead(
                hidden,
                self.num_classes,
                max_queries=16,
                hidden_channels=hidden,
                token_grid=8,
            )
        elif dynamic_head_variant == "fixed":
            self.dynamic_output = None
        else:
            raise ValueError(f"unknown dynamic head variant: {dynamic_head_variant}")
        self.dynamic_head_variant = dynamic_head_variant

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        """Load legacy checkpoints whose dynamic router did not exist yet."""

        current = self.state_dict()
        compatible = {
            name: value
            for name, value in state_dict.items()
            if name in current and current[name].shape == value.shape
        }
        result = super().load_state_dict(compatible, strict=False, assign=assign)
        disallowed_missing = [
            name
            for name in result.missing_keys
            if not name.startswith("dynamic_output.")
        ]
        if strict and (disallowed_missing or result.unexpected_keys):
            raise RuntimeError(
                "incompatible pixel head state: "
                f"missing={disallowed_missing}, "
                f"unexpected={list(result.unexpected_keys)}"
            )
        return result

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = torch.cat(
            [branch(features) for branch in self.multi_scale],
            dim=1,
        )
        logits = self.mix(hidden)
        query_diagnostics = None
        if self.dynamic_output is not None:
            dynamic_result = self.dynamic_output(hidden)
            if isinstance(dynamic_result, tuple):
                dynamic_residual, query_diagnostics = dynamic_result
            else:
                dynamic_residual = dynamic_result
            logits = logits + dynamic_residual
        probability = logits.softmax(dim=1)
        entropy = -(
            probability.clamp_min(1.0e-8)
            * probability.clamp_min(1.0e-8).log()
        ).sum(dim=1, keepdim=True)
        reliability = 1.0 - entropy / max(1.0e-6, math.log(float(self.num_classes)))
        room_ids = self.room_ids.to(logits.device)
        room_logits = (
            logits[:, room_ids]
            if room_ids.numel()
            else logits.new_zeros((logits.shape[0], 0, *logits.shape[-2:]))
        )
        return {
            "logits": logits,
            "probability": probability,
            "room_logits": room_logits,
            "reliability": reliability,
            "multi_scale_weights": (
                getattr(self.dynamic_output, "last_scale_weights_raw", None)
                if self.dynamic_output is not None
                else None
            ),
            "dynamic_query_diagnostics": query_diagnostics,
        }


def apply_distribution_prior(
    channel_logits: torch.Tensor,
    distribution: dict[str, torch.Tensor],
    *,
    scale: float = 0.1,
    room_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add a class-wise log-prior to room logits.

    A reliability map is a single channel per pixel; adding it to every class
    cancels inside the softmax, so it is intentionally an identity.  Passing a
    ``[B,K,H,W]`` ``room_prior`` (for example the semantic-element room
    probability) produces a real, class-wise gate.
    """

    if room_prior is not None:
        prior = room_prior.to(channel_logits.dtype).clamp_min(1.0e-6).log()
        if prior.shape != channel_logits.shape:
            raise ValueError(
                "room_prior shape must match channel_logits: "
                f"{prior.shape} vs {channel_logits.shape}"
            )
        return channel_logits + float(scale) * prior
    reliability = distribution["reliability"].to(channel_logits.dtype)
    if reliability.shape[-2:] != channel_logits.shape[-2:]:
        raise ValueError("reliability spatial size must match channel logits")
    return channel_logits


def pixel_distribution_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    class_weights: torch.Tensor | None = None,
    ce_weight: float = 1.0,
    room_iou_weight: float = 0.5,
    tversky_weight: float = 0.2,
    family_iou_weight: float = 0.0,
    boundary_weight: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dense CE plus a direct differentiable surrogate for room macro IoU."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    valid = footprint[:, 0].bool()
    ce_pixels = F.cross_entropy(
        logits,
        target.long(),
        weight=class_weights,
        reduction="none",
    )
    ce_per_sample = (ce_pixels * valid).flatten(1).sum(dim=1) / valid.flatten(
        1
    ).sum(dim=1).clamp_min(1)
    if sample_weights is None:
        sample_weights = torch.ones(
            logits.shape[0],
            device=logits.device,
            dtype=logits.dtype,
        )
    else:
        sample_weights = sample_weights.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        if sample_weights.shape != (logits.shape[0],):
            raise ValueError("sample_weights must have shape [batch]")
    normalizer = sample_weights.sum().clamp_min(1.0e-6)
    ce = (ce_per_sample * sample_weights).sum() / normalizer
    probability = logits.softmax(dim=1)
    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=logits.device,
        dtype=torch.long,
    )
    room_probability = probability[:, room_ids]
    lookup = torch.full(
        (taxonomy.num_classes,),
        -1,
        device=logits.device,
        dtype=torch.long,
    )
    lookup[room_ids] = torch.arange(len(room_ids), device=logits.device)
    room_target = lookup[target]
    one_hot = F.one_hot(
        room_target.clamp_min(0),
        len(room_ids),
    ).permute(0, 3, 1, 2).float()
    room_valid = (room_target >= 0)[:, None] & footprint.bool()
    one_hot = one_hot * room_valid
    room_probability = room_probability * room_valid
    intersection = (room_probability * one_hot).sum(dim=(2, 3))
    predicted_mass = room_probability.sum(dim=(2, 3))
    target_mass = one_hot.sum(dim=(2, 3))
    union = predicted_mass + target_mass - intersection
    present = target_mass > 0
    iou = intersection / union.clamp_min(1.0)
    soft_iou_per_sample = 1.0 - (
        iou.masked_fill(~present, 0.0).sum(dim=1)
        / present.sum(dim=1).clamp_min(1)
    )
    soft_iou_loss = (soft_iou_per_sample * sample_weights).sum() / normalizer
    false_positive = (predicted_mass - intersection).clamp_min(0)
    false_negative = (target_mass - intersection).clamp_min(0)
    tversky = (intersection + 1.0) / (
        intersection
        + 0.3 * false_positive
        + 0.7 * false_negative
        + 1.0
    )
    tversky_per_sample = 1.0 - (
        tversky.masked_fill(~present, 0.0).sum(dim=1)
        / present.sum(dim=1).clamp_min(1)
    )
    tversky_loss = (tversky_per_sample * sample_weights).sum() / normalizer
    eroded_probability = -F.max_pool2d(
        -room_probability,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    predicted_boundary = (room_probability - eroded_probability).clamp_min(0.0)
    eroded_target = -F.max_pool2d(
        -one_hot,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    target_boundary = (one_hot - eroded_target).clamp_min(0.0)
    boundary_intersection = (predicted_boundary * target_boundary).sum(
        dim=(2, 3)
    )
    boundary_denominator = predicted_boundary.sum(dim=(2, 3)) + (
        target_boundary.sum(dim=(2, 3))
    )
    boundary_dice = (2.0 * boundary_intersection + 1.0) / (
        boundary_denominator + 1.0
    )
    boundary_per_sample = 1.0 - (
        boundary_dice.masked_fill(~present, 0.0).sum(dim=1)
        / present.sum(dim=1).clamp_min(1)
    )
    boundary_loss = (boundary_per_sample * sample_weights).sum() / normalizer
    family_weighted_loss = ce * 0.0
    family_components: dict[str, torch.Tensor] = {}
    if family_iou_weight > 0:
        groups = {
            "wall": (taxonomy.class_id("wall"),),
            "opening": tuple(taxonomy.opening_ids),
            "room": tuple(taxonomy.room_ids),
            "furniture": tuple(taxonomy.furniture_ids),
            "support": (taxonomy.class_id("structure"),),
        }
        one_hot_target = F.one_hot(
            target.long(),
            num_classes=taxonomy.num_classes,
        ).permute(0, 3, 1, 2).float()
        valid_float = valid[:, None].float()
        predicted_masked = probability * valid_float
        target_masked = one_hot_target * valid_float
        intersection = (predicted_masked * target_masked).sum(dim=(0, 2, 3))
        predicted_mass = predicted_masked.sum(dim=(0, 2, 3))
        target_mass = target_masked.sum(dim=(0, 2, 3))
        class_iou = (intersection + 1.0) / (
            predicted_mass + target_mass - intersection + 1.0
        )
        present = target_mass > 0
        family_losses = []
        for name, class_ids in groups.items():
            class_ids = tuple(class_id for class_id in class_ids if class_id > 0)
            if not class_ids:
                continue
            indices = torch.tensor(class_ids, device=logits.device)
            selected_present = present.index_select(0, indices)
            if selected_present.any():
                value = 1.0 - class_iou.index_select(
                    0,
                    indices,
                )[selected_present].mean()
                family_losses.append(value)
                family_components[f"pixel_{name}_iou_loss"] = value.detach()
        if family_losses:
            family_weighted_loss = torch.stack(family_losses).mean()
    total = (
        float(ce_weight) * ce
        + float(room_iou_weight) * soft_iou_loss
        + float(tversky_weight) * tversky_loss
        + float(family_iou_weight) * family_weighted_loss
        + float(boundary_weight) * boundary_loss
    )
    return total, {
        "pixel_distribution_ce": ce.detach(),
        "room_soft_iou_loss": soft_iou_loss.detach(),
        "room_tversky_loss": tversky_loss.detach(),
        "room_boundary_loss": boundary_loss.detach(),
        "pixel_distribution_total": total.detach(),
        "ce_term": ce,
        "room_iou_term": soft_iou_loss,
        "tversky_term": tversky_loss,
        "boundary_term": boundary_loss,
        "family_iou_term": family_weighted_loss,
        **family_components,
    }


__all__ = [
    "PixelDistributionHead",
    "apply_distribution_prior",
    "pixel_distribution_loss",
]

"""Differentiable object rendering adapted from FloorplanGAN (MIT).

The upstream ``renderer_g2v`` renders rectangle primitives with a soft edge
and max-projects class scores over instances.  The same max-projection
structure is applied to instance query masks here, which keeps the furniture
placement loss differentiable through both class and mask logits while the
inference path keeps its hard rendering.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_object_render(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    footprint: torch.Tensor,
    *,
    temperature: float = 1.0,
    active_queries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render queries to a ``[B,C+1,H,W]`` differentiable class mixture."""

    if class_logits.ndim != 3 or mask_logits.ndim != 4:
        raise ValueError("class_logits must be [B,Q,K+1] and masks [B,Q,H,W]")
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if mask_logits.shape[-2:] != footprint.shape[-2:]:
        mask_logits = F.interpolate(
            mask_logits,
            size=footprint.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    temperature = max(1.0e-3, float(temperature))
    class_probability = (class_logits / temperature).softmax(dim=-1)[..., :-1]
    mask_probability = mask_logits.sigmoid()
    per_query = mask_probability[:, :, None] * class_probability[:, :, :, None, None]
    if active_queries is not None:
        if active_queries.shape != class_probability.shape[:2]:
            raise ValueError("active_queries must match [B,Q]")
        if not bool(active_queries.any(dim=1).all()):
            raise ValueError("each sample must retain one active query")
        per_query = per_query * active_queries[:, :, None, None, None].to(
            per_query.dtype
        )
    class_scores = per_query.amax(dim=1)
    foreground = class_scores.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    rendered = torch.cat([1.0 - foreground, class_scores], dim=1)
    return rendered * footprint


def furniture_vector_loss(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    target_labels: torch.Tensor,
    class_ids: tuple[int, ...],
    footprint: torch.Tensor,
    *,
    temperature: float = 1.0,
    smoothness: float = 1.0e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Class-mixture cross entropy on the differentiable rendered furniture."""

    rendered = soft_object_render(
        class_logits,
        mask_logits,
        footprint,
        temperature=temperature,
    )
    normalized = (rendered + float(smoothness)).clamp_min(1.0e-6)
    normalized = normalized / normalized.sum(dim=1, keepdim=True)
    valid = footprint[:, 0].bool()
    index = torch.full_like(target_labels, -1, dtype=torch.long)
    for class_index, class_id in enumerate(class_ids):
        index[target_labels == int(class_id)] = class_index + 1
    supervised = valid & (index >= 0)
    if not supervised.any():
        zero = rendered.sum() * 0.0
        return zero, {
            "furniture_vector_loss": zero.detach(),
            "furniture_vector_supervised_pixels": rendered.new_tensor(0.0),
        }
    log_probability = normalized.permute(0, 2, 3, 1)
    loss = F.nll_loss(
        log_probability[supervised],
        index[supervised],
    )
    return loss, {
        "furniture_vector_loss": loss.detach(),
        "furniture_vector_supervised_pixels": supervised.sum().to(rendered.dtype),
    }


def furniture_containment_loss(
    mask_logits: torch.Tensor,
    room_support: torch.Tensor,
) -> torch.Tensor:
    """Penalize predicted furniture probability outside any room."""

    if mask_logits.ndim != 4:
        raise ValueError("mask_logits must be [B,Q,H,W]")
    if room_support.ndim == 4:
        room_support = room_support[:, 0]
    if room_support.ndim != 3:
        raise ValueError("room_support must be [B,H,W] or [B,1,H,W]")
    probability = mask_logits.sigmoid()
    if probability.shape[-2:] != room_support.shape[-2:]:
        probability = F.interpolate(
            probability,
            size=room_support.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    outside = probability * (1.0 - room_support[:, None].float())
    return outside.mean()


__all__ = [
    "furniture_containment_loss",
    "furniture_vector_loss",
    "soft_object_render",
]

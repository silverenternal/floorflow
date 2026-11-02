"""GPU-safe attention mask normalization shared by residual heads."""

from __future__ import annotations

import math

import torch

from floorflow.geometry.deterministic_pooling import fixed_grid_pool


def unmask_first_empty_row(padding: torch.Tensor) -> torch.Tensor:
    """Keep one row visible when an entire batch item is padding.

    Fully masked attention rows are undefined in some backends. The previous
    implementation synchronized on `padding.all(dim=1).any()` before patching
    the first row. This version uses only device-side tensor operations.
    """

    if padding.ndim != 2:
        raise ValueError("padding must be [B,N]")
    if padding.shape[1] == 0:
        return padding
    empty = padding.all(dim=1)
    first = torch.zeros_like(padding)
    first[:, 0] = empty
    return padding & ~first


def soft_pixel_attention_bias(
    mask_logits: torch.Tensor | None,
    size: tuple[int, int],
    *,
    floor: float = 0.05,
    scale: float = 2.0,
) -> torch.Tensor | None:
    """Turn a previous channel mask into a recoverable additive attention bias.

    A hard boolean mask makes a false negative irreversible in later layers.
    Keeping a finite floor lets uncertain pixels re-enter channel attention while
    still suppressing confidently owned pixels.
    """

    if mask_logits is None:
        return None
    if not 0.0 < floor < 1.0:
        raise ValueError("attention mask floor must be between zero and one")
    if scale < 0.0:
        raise ValueError("attention mask scale must be non-negative")
    probability = fixed_grid_pool(mask_logits.sigmoid(), size)
    weight = (probability.clamp_min(floor) / floor).log() * float(scale)
    return weight.flatten(2)


def soft_owner_attention_bias(
    mask_logits: torch.Tensor,
    *,
    floor: float = 0.05,
    scale: float = 2.0,
) -> torch.Tensor:
    """Suppress confident competing room cores with a finite attention bias."""

    if mask_logits.ndim not in {3, 4}:
        raise ValueError("owner mask logits must be [B,Q,N] or [B,Q,H,W]")
    if not 0.0 < floor < 1.0:
        raise ValueError("attention mask floor must be between zero and one")
    if scale < 0.0:
        raise ValueError("attention mask scale must be non-negative")
    probability = mask_logits.softmax(dim=1).detach()
    confidence, owner = probability.max(dim=1)
    query_ids = torch.arange(
        mask_logits.shape[1],
        device=mask_logits.device,
    )
    if mask_logits.ndim == 3:
        competing_core = (
            owner[:, None] != query_ids[None, :, None]
        ) * confidence[:, None]
        return -float(scale) * (
            competing_core / float(floor)
        ).clamp_min(1.0).log()
    competing_core = (
        (owner[:, None] != query_ids[None, :, None, None])
        * confidence[:, None]
    )
    penalty = -float(scale) * (
        competing_core / float(floor)
    ).clamp_min(1.0).log()
    return penalty.flatten(2)


def soft_presence_attention_bias(
    padding: torch.Tensor,
    *,
    floor: float = 0.05,
    scale: float = 2.0,
) -> torch.Tensor:
    """Represent padded keys with the same finite floor as spatial masks."""

    if not 0.0 < floor < 1.0:
        raise ValueError("attention mask floor must be between zero and one")
    if scale < 0.0:
        raise ValueError("attention mask scale must be non-negative")
    return padding.to(torch.float32) * (
        math.log(float(floor)) * float(scale)
    )


__all__ = [
    "soft_owner_attention_bias",
    "soft_pixel_attention_bias",
    "soft_presence_attention_bias",
    "unmask_first_empty_row",
]

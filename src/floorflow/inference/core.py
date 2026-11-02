"""Reusable non-selecting inference helpers."""

from __future__ import annotations

import torch


@torch.no_grad()
def semantic_endpoint_labels(
    model,
    reference: torch.Tensor,
    footprint: torch.Tensor,
    floor_ids: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros(
        reference.shape[0],
        model.target_channels,
        *reference.shape[-2:],
        device=reference.device,
        dtype=footprint.dtype,
    )
    features = model._encode_features(
        zero,
        torch.zeros(reference.shape[0], device=reference.device, dtype=footprint.dtype),
        footprint,
        floor_ids=floor_ids,
    )
    logits = model.semantic_endpoint(features)
    return torch.cat([torch.zeros_like(logits[:, :1]), logits], dim=1).argmax(dim=1)

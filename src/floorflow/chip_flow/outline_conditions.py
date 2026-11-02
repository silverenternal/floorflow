"""Normalized outline conditioning helpers for chip flow models."""

from __future__ import annotations

import torch


def outline_condition_from_boxes(
    boxes: torch.Tensor,
    areas: torch.Tensor,
    margin: torch.Tensor | float = 0.0,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Build normalized centered outline states from batched GT boxes."""

    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    x0 = boxes[..., 0].amin(dim=1)
    y0 = boxes[..., 1].amin(dim=1)
    x1 = (boxes[..., 0] + boxes[..., 2]).amax(dim=1)
    y1 = (boxes[..., 1] + boxes[..., 3]).amax(dim=1)
    width = x1 - x0
    height = y1 - y0
    margin_value = torch.as_tensor(
        margin,
        dtype=boxes.dtype,
        device=boxes.device,
    )
    if margin_value.ndim == 0:
        margin_value = margin_value.expand(boxes.shape[0])
    if margin_value.shape != (boxes.shape[0],):
        raise ValueError("outline margin must be scalar or one value per case")
    if validate:
        invalid = ~torch.isfinite(margin_value).all() | (
            margin_value < 0.0
        ).any()
        if invalid:
            raise ValueError("outline margin must be finite and non-negative")
    width = width * (1.0 + 2.0 * margin_value)
    height = height * (1.0 + 2.0 * margin_value)
    return torch.stack(
        [
            (x0 + x1) * 0.5 / scale,
            (y0 + y1) * 0.5 / scale,
            width / scale,
            height / scale,
        ],
        dim=-1,
    )


def outline_condition_from_hw(
    outline: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    """Build normalized centered outline states from explicit ``(W, H)``."""

    outline = outline.to(device=areas.device, dtype=areas.dtype)
    if outline.ndim != 2 or outline.shape != (areas.shape[0], 2):
        raise ValueError("outline must have shape (batch, 2)")
    invalid = ~torch.isfinite(outline).all() | (outline <= 0.0).any()
    if invalid:
        raise ValueError("outline dimensions must be finite and positive")
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    return torch.stack(
        [
            outline[:, 0] / (2.0 * scale),
            outline[:, 1] / (2.0 * scale),
            outline[:, 0] / scale,
            outline[:, 1] / scale,
        ],
        dim=-1,
    )


__all__ = ["outline_condition_from_boxes", "outline_condition_from_hw"]

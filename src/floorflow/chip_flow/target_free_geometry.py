"""Target-free geometry metrics for bounded portfolio selection."""

from __future__ import annotations

import torch


def target_free_geometry_metrics(
    boxes: torch.Tensor,
    areas: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
    n_b2b: int,
    n_p2b: int,
) -> dict[str, float]:
    """Return absolute HPWL and outline compactness without target boxes."""

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if areas.shape != (boxes.shape[0],):
        raise ValueError("areas must match box count")
    scale = areas.sum().sqrt().clamp_min(1.0e-6)
    centers = boxes[:, :2] + boxes[:, 2:] * 0.5
    b2b_value = boxes.new_zeros(())
    if n_b2b > 0 and b2b.numel():
        edges = b2b[:n_b2b]
        left = edges[:, 0].long().clamp(0, boxes.shape[0] - 1)
        right = edges[:, 1].long().clamp(0, boxes.shape[0] - 1)
        b2b_value = (
            (centers[left] - centers[right]).abs().sum(dim=-1)
            * edges[:, 2].clamp_min(0.0)
        ).sum()
    p2b_value = boxes.new_zeros(())
    if n_p2b > 0 and p2b.numel() and pins.numel():
        edges = p2b[:n_p2b]
        pin_id = edges[:, 0].long().clamp(0, pins.shape[0] - 1)
        block_id = edges[:, 1].long().clamp(0, boxes.shape[0] - 1)
        p2b_value = (
            (pins[pin_id].to(boxes.device) - centers[block_id])
            .abs()
            .sum(dim=-1)
            * edges[:, 2].clamp_min(0.0)
        ).sum()
    bbox_width = (
        boxes[:, 0].amax() - boxes[:, 0].amin()
    ).clamp_min(1.0e-6)
    bbox_height = (
        (boxes[:, 1] + boxes[:, 3]).amax() - boxes[:, 1].amin()
    ).clamp_min(1.0e-6)
    occupied = (boxes[:, 2] * boxes[:, 3]).sum()
    compactness = occupied / (bbox_width * bbox_height)
    return {
        "b2b_hpwl": float(b2b_value / scale),
        "p2b_hpwl": float(p2b_value / scale),
        "outline_compactness": float(compactness),
        "outline_aspect": float(torch.log(bbox_width / bbox_height)),
    }


__all__ = ["target_free_geometry_metrics"]

"""Deterministic fixed-grid spatial pooling for CUDA training."""

from __future__ import annotations

import torch.nn.functional as F


def fixed_grid_pool(
    value,
    size: tuple[int, int],
):
    """Pool to an exact grid without adaptive-pool CUDA backward."""

    target_h, target_w = int(size[0]), int(size[1])
    height, width = value.shape[-2:]
    if (height, width) == (target_h, target_w):
        return value
    if target_h > height or target_w > width:
        return F.interpolate(
            value,
            size=(target_h, target_w),
            mode="nearest",
        )
    stride_h = max(1, height // target_h)
    stride_w = max(1, width // target_w)
    kernel_h = height - stride_h * (target_h - 1)
    kernel_w = width - stride_w * (target_w - 1)
    return F.avg_pool2d(
        value,
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
    )


__all__ = ["fixed_grid_pool"]

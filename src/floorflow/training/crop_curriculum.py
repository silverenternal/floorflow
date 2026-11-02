"""Scale-crop augmentation for recursive floorplan subregion decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_crop_and_resize(
    target: torch.Tensor,
    footprint: torch.Tensor,
    *,
    probability: float = 0.5,
    minimum_fraction: float = 0.35,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly crop each sample and resize it back to the model resolution."""

    if target.ndim != 3 or footprint.ndim != 3:
        raise ValueError("target and footprint must be [B,H,W]")
    if target.shape != footprint.shape:
        raise ValueError("target and footprint shapes differ")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not 0.0 < float(minimum_fraction) <= 1.0:
        raise ValueError("minimum_fraction must be in (0, 1]")
    output_target = target.clone()
    output_footprint = footprint.clone()
    height, width = target.shape[-2:]
    for index in range(target.shape[0]):
        if float(torch.rand((), generator=generator)) >= float(probability):
            continue
        fraction = float(
            torch.empty(())
            .uniform_(
                float(minimum_fraction),
                1.0,
                generator=generator,
            )
        )
        crop_height = max(8, round(height * fraction))
        crop_width = max(8, round(width * fraction))
        top = int(
            torch.randint(
                0,
                max(1, height - crop_height + 1),
                (1,),
                generator=generator,
            )
        )
        left = int(
            torch.randint(
                0,
                max(1, width - crop_width + 1),
                (1,),
                generator=generator,
            )
        )
        target_crop = target[
            index : index + 1,
            top : top + crop_height,
            left : left + crop_width,
        ]
        footprint_crop = footprint[
            index : index + 1,
            top : top + crop_height,
            left : left + crop_width,
        ]
        output_target[index : index + 1] = F.interpolate(
            target_crop.float()[:, None],
            size=(height, width),
            mode="nearest",
        )[:, 0].to(target.dtype)
        output_footprint[index : index + 1] = F.interpolate(
            footprint_crop.float()[:, None],
            size=(height, width),
            mode="nearest",
        )[:, 0].to(footprint.dtype)
    return output_target, output_footprint


__all__ = ["random_crop_and_resize"]

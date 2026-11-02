"""Runtime helpers for raster-conditioned room controls."""

from __future__ import annotations

import numpy as np
import torch

from floorflow.data.raster_control import (
    image_normalization,
    raster_control_image,
)
from floorflow.heads.raster_control_bank import load_raster_room_model


def raster_control_batch(
    target: torch.Tensor,
    footprint: torch.Tensor,
    control_image: torch.Tensor | None,
    taxonomy,
    device: torch.device,
) -> torch.Tensor:
    if control_image is not None:
        return control_image.to(device)
    images = [
        image_normalization(
            raster_control_image(
                row.detach().cpu().numpy(),
                mask.detach().cpu().numpy(),
                taxonomy,
            )
        )
        for row, mask in zip(target, footprint[:, 0], strict=True)
    ]
    return torch.from_numpy(np.stack(images)).to(device)


__all__ = ["load_raster_room_model", "raster_control_batch"]

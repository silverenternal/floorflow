"""Fuse missing serialized control regions with a frozen fallback prior."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

SPATIAL_CONTROL_KEYS = (
    "room_logits",
    "structure",
    "furniture",
    "support",
)


def fuse_controls(
    primary: dict[str, torch.Tensor],
    fallback: dict[str, torch.Tensor],
    validity_masks: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    fused = {}
    for key, value in primary.items():
        if key == "program":
            fused[key] = value.clone()
            continue
        mask_name = f"valid_{key}"
        if mask_name not in validity_masks:
            raise ValueError(f"missing validity mask: {mask_name}")
        if key not in fallback:
            raise ValueError(f"fallback control missing: {key}")
        if value.shape != fallback[key].shape:
            raise ValueError(
                f"{key} shape mismatch: {value.shape} vs {fallback[key].shape}"
            )
        mask = torch.from_numpy(validity_masks[mask_name].astype(bool))
        output = value.clone()
        output[..., ~mask] = fallback[key][..., ~mask]
        fused[key] = output
    return fused


def inpaint_controls(
    primary: dict[str, torch.Tensor],
    validity_masks: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    output = {}
    for key, value in primary.items():
        if key == "program":
            output[key] = value.clone()
            continue
        mask_name = f"valid_{key}"
        if mask_name not in validity_masks:
            raise ValueError(f"missing validity mask: {mask_name}")
        valid = torch.from_numpy(validity_masks[mask_name].astype(bool))
        array = value.cpu().numpy().copy()
        indices = ndimage.distance_transform_edt(
            ~valid.numpy(),
            return_distances=False,
            return_indices=True,
        )
        array = array[..., indices[0], indices[1]]
        output[key] = torch.from_numpy(array).to(device=value.device, dtype=value.dtype)
    return output


def nearest_fill_batch(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor fill a batched control tensor with one EDT per sample."""

    if valid.ndim != 3 or valid.shape[0] != value.shape[0]:
        raise ValueError("valid must be [B,H,W] matching the control batch")
    outputs = []
    for index in range(value.shape[0]):
        valid_numpy = valid[index].detach().cpu().numpy().astype(bool)
        array = value[index].detach().cpu().numpy().copy()
        indices = ndimage.distance_transform_edt(
            ~valid_numpy,
            return_distances=False,
            return_indices=True,
        )
        array = array[..., indices[0], indices[1]]
        outputs.append(
            torch.from_numpy(array).to(device=value.device, dtype=value.dtype)
        )
    return torch.stack(outputs)


__all__ = [
    "SPATIAL_CONTROL_KEYS",
    "fuse_controls",
    "inpaint_controls",
    "nearest_fill_batch",
]

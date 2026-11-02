"""Deterministic structural perturbations for serialized controls."""

from __future__ import annotations

import math

import torch


def spatial_block(
    shape: tuple[int, int],
    fraction: float,
    *,
    seed: int,
) -> torch.Tensor:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("drop fraction must be in [0, 1]")
    height, width = shape
    if fraction == 0.0:
        return torch.zeros(shape, dtype=torch.bool)
    side = max(1, round(math.sqrt(fraction) * min(height, width)))
    generator = torch.Generator().manual_seed(int(seed))
    top = int(torch.randint(0, max(1, height - side + 1), (1,), generator=generator))
    left = int(torch.randint(0, max(1, width - side + 1), (1,), generator=generator))
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[top : top + side, left : left + side] = True
    return mask


def corrupt_controls(
    controls: dict[str, torch.Tensor],
    fraction: float,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    output = {}
    for key, value in controls.items():
        if key == "program":
            output[key] = value.clone()
            continue
        mask = spatial_block(tuple(value.shape[-2:]), fraction, seed=seed)
        corrupted = value.clone()
        corrupted[..., mask] = 0.0
        output[key] = corrupted
    return output


def control_validity_masks(
    controls: dict[str, torch.Tensor],
    fraction: float,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    return {
        f"valid_{key}": ~spatial_block(
            tuple(value.shape[-2:]),
            fraction,
            seed=seed,
        )
        for key, value in controls.items()
        if key != "program"
    }


def random_spatial_block_mask(
    batch_size: int,
    shape: tuple[int, int],
    fraction: torch.Tensor | float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    masks = []
    for index in range(int(batch_size)):
        value = (
            float(fraction[index])
            if isinstance(fraction, torch.Tensor)
            else float(fraction)
        )
        masks.append(
            spatial_block(
                shape,
                value,
                seed=int(
                    torch.randint(
                        0,
                        2**31 - 1,
                        (1,),
                        generator=generator,
                    )
                ),
            )
        )
    return torch.stack(masks)


__all__ = [
    "control_validity_masks",
    "corrupt_controls",
    "random_spatial_block_mask",
    "spatial_block",
]

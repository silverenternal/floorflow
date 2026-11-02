"""Exterior-contour image loading for standalone inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def fill_enclosed_regions(mask: np.ndarray) -> np.ndarray:
    """Fill holes enclosed by a binary outline."""

    mask = mask.astype(np.uint8)
    if not mask.any():
        return mask
    height, width = mask.shape
    padded = np.zeros((height + 4, width + 4), dtype=np.uint8)
    padded[2 : 2 + height, 2 : 2 + width] = mask
    image = Image.fromarray(np.ascontiguousarray(padded * 255), mode="L").copy()
    ImageDraw.floodfill(image, (0, 0), 128)
    outside = np.asarray(image, dtype=np.uint8) == 128
    return (~outside).astype(np.uint8)[2 : 2 + height, 2 : 2 + width]


def load_footprint(
    path: str | Path,
    *,
    image_size: int = 256,
    foreground: str = "white",
    fill_outline: bool = False,
    margin: int | None = None,
) -> np.ndarray:
    """Load a contour image and fit it into the model's square condition frame."""

    if foreground not in {"white", "black"}:
        raise ValueError("foreground must be 'white' or 'black'")
    image = Image.open(path).convert("L")
    values = np.asarray(image, dtype=np.uint8)
    mask = (values > 127).astype(np.uint8) if foreground == "white" else (values < 128).astype(np.uint8)
    if fill_outline:
        mask = fill_enclosed_regions(mask)
    if not mask.any():
        raise ValueError(f"no {foreground} foreground pixels found in {path}")

    ys, xs = np.nonzero(mask)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    crop = Image.fromarray(mask[top:bottom, left:right] * 255, mode="L")
    margin = max(2, image_size // 64) if margin is None else max(0, int(margin))
    available = max(1, image_size - 2 * margin)
    crop = ImageOps.contain(crop, (available, available), method=Image.Resampling.NEAREST)
    canvas = Image.new("L", (image_size, image_size), 0)
    offset = ((image_size - crop.width) // 2, (image_size - crop.height) // 2)
    canvas.paste(crop, offset)
    return (np.asarray(canvas, dtype=np.uint8) > 127).astype(np.uint8)

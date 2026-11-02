"""Small NumPy/Pillow visualizers for training and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from floorflow.data.classes import CLASS_COLORS
from floorflow.data.taxonomy import FloorplanTaxonomy


def colorize(labels: np.ndarray, taxonomy: FloorplanTaxonomy | None = None) -> np.ndarray:
    if taxonomy is None:
        colors = np.asarray(CLASS_COLORS, dtype=np.uint8)
    else:
        colors = np.asarray([taxonomy.color(index) for index in range(taxonomy.num_classes)], dtype=np.uint8)
    clipped = np.clip(labels.astype(np.int64), 0, len(colors) - 1)
    return colors[clipped]


def make_comparison_panel(
    footprint: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    title: str = "",
    taxonomy: FloorplanTaxonomy | None = None,
) -> Image.Image:
    panels = [
        np.repeat((footprint[..., None] * 255).astype(np.uint8), 3, axis=2),
        colorize(target, taxonomy),
        colorize(prediction, taxonomy),
    ]
    height, width = target.shape
    gap = 8
    header = 24 if title else 0
    canvas = Image.new("RGB", (width * 3 + gap * 2, height + header), "white")
    for index, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (index * (width + gap), header))
    if title:
        ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def save_comparison_panel(
    path: str | Path,
    footprint: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    title: str = "",
    taxonomy: FloorplanTaxonomy | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_comparison_panel(footprint, target, prediction, title=title, taxonomy=taxonomy).save(path)

"""Convert FloorFlow dense predictions into Raster2Seq-style colored inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.raster2seq_external import (
    render_raster2seq_official_style,
)
from floorflow.heads.structured_objects import (
    compose_structured_objects,
    object_group_predictions,
)


@dataclass(frozen=True)
class DenseColorizedInput:
    labels: np.ndarray
    image: np.ndarray
    mode: str
    wall_pixels: int
    opening_pixels: int
    furniture_pixels: int


@torch.inference_mode()
def dense_colorized_raster2seq_input(
    outputs: dict[str, torch.Tensor],
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    mode: str = "rooms_walls",
    wall_probability_threshold: float = 0.7,
    opening_probability_threshold: float = 0.5,
    furniture_probability_threshold: float = 0.5,
) -> DenseColorizedInput:
    """Compose target-free dense predictions into one Raster2Seq input."""

    if mode not in {"rooms", "rooms_walls", "full_structure"}:
        raise ValueError(f"unknown dense colorization mode: {mode}")
    if footprint.ndim != 3 or footprint.shape[0] != 1:
        raise ValueError("expected one footprint map [1,H,W]")
    room_ids = torch.tensor(taxonomy.room_ids, device=footprint.device)
    room_logits = outputs.get("fused_logits", outputs["room_logits"])
    labels = room_ids[room_logits.argmax(dim=1)] * footprint.long()
    if mode == "rooms":
        objects = {
            "wall": torch.zeros_like(labels),
            "opening": torch.zeros_like(labels),
            "furniture": torch.zeros_like(labels),
        }
    else:
        wall_logit_threshold = float(
            torch.logit(torch.tensor(float(wall_probability_threshold)))
        )
        objects = object_group_predictions(
            outputs,
            taxonomy,
            opening_threshold=float(
                torch.logit(
                    torch.tensor(float(opening_probability_threshold))
                )
            ),
            furniture_threshold=float(
                torch.logit(
                    torch.tensor(float(furniture_probability_threshold))
                )
            ),
            wall_threshold=wall_logit_threshold,
        )
        if mode == "rooms_walls":
            objects["opening"] = torch.zeros_like(objects["opening"])
            objects["furniture"] = torch.zeros_like(objects["furniture"])
    labels = compose_structured_objects(
        labels,
        objects,
        footprint,
        taxonomy,
    )
    image = render_raster2seq_official_style(
        labels[0].cpu().numpy(),
        taxonomy,
    )
    return DenseColorizedInput(
        labels=labels[0].cpu().numpy(),
        image=image,
        mode=mode,
        wall_pixels=int(objects["wall"].sum()),
        opening_pixels=int((objects["opening"] != 0).sum()),
        furniture_pixels=int((objects["furniture"] != 0).sum()),
    )


__all__ = [
    "DenseColorizedInput",
    "dense_colorized_raster2seq_input",
]

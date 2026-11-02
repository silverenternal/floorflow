"""Deterministic wall readout from final-stage canonical CAD layers."""

from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import (
    ReadoutWall,
    WallSkeletonPolicy,
    extract_wall_segments,
)
from floorflow.inference.canonical_layer_composition import (
    canonical_union_labels,
)


@torch.no_grad()
def final_stage_wall_segments(
    layer_output: dict[str, object],
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    visible_wall_boundary_width: int = 7,
    close_kernel: int = 5,
    simplification_fraction: float = 0.010,
    minimum_segment_length: float = 2.5,
) -> list[list[ReadoutWall]]:
    """Read source-5 walls using the canonical deployed layer contract."""

    layers = layer_output["cad_layers"]
    target_size = tuple(footprint.shape[-2:])
    resized = layers.resized(target_size)
    labels = canonical_union_labels(
        resized.render(footprint, taxonomy),
        taxonomy,
        boundary_width=int(visible_wall_boundary_width),
    )
    policy = WallSkeletonPolicy(
        close_kernel=int(close_kernel),
        simplification_fraction=float(simplification_fraction),
        minimum_segment_length=float(minimum_segment_length),
    )
    return [
        extract_wall_segments(
            label.detach().cpu().numpy(),
            taxonomy,
            wall_policy=policy,
        )
        for label in labels
    ]


__all__ = ["final_stage_wall_segments"]

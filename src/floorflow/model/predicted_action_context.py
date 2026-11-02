"""Build raw CAD action inputs from predicted floorplan controls."""

from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy


def predicted_action_inputs(
    taxonomy: FloorplanTaxonomy,
    controls: dict[str, torch.Tensor],
    footprint: torch.Tensor,
) -> dict[str, torch.Tensor]:
    room_count = len(taxonomy.room_ids)
    structure = controls["structure"].float()
    furniture = controls["furniture"].float()
    junction = controls.get("junction_probability")
    if junction is not None and junction.ndim == 3:
        junction = junction.float()[:, None]
    if junction is None:
        junction = structure[:, room_count : room_count + 1]
    return {
        "envelope": (
            footprint.float()[:, None]
            if footprint.ndim == 3
            else footprint.float()
        ),
        "program": controls["program"].float(),
        "rooms": controls["room_logits"].float(),
        "walls": structure[:, room_count : room_count + 1],
        "junctions": junction,
        "openings": structure[
            :,
            room_count + 1 : room_count + 2 + len(taxonomy.opening_ids),
        ],
        "furniture": furniture[
            :, : len(taxonomy.furniture_ids) + 1
        ],
        "validation": controls["support"].float(),
    }


def maybe_detach(value: torch.Tensor, enabled: bool) -> torch.Tensor:
    return value.detach() if enabled else value


__all__ = ["maybe_detach", "predicted_action_inputs"]

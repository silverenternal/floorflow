"""Adapter from room-layout latents to Trinity's closed-form refiner."""

from __future__ import annotations

import numpy as np
import torch

from trinity.decode import z_to_xywh
from trinity.floorplan.legalize.scale_pack import scale_pack_legalize
from trinity.sampling.refine_closed import refine_closed
from trinity.sampling.refine_constraint import RefineCase
from trinity.floorplan.types import FloorplanInstance, Placement

from floorflow.representation.trinity_latent import TrinityRoomBatch
from floorflow.representation.trinity_latent import (
    render_room_boxes,
)
from floorflow.representation.trinity_polygons import room_part_batch_from_labels
from floorflow.data.taxonomy import FloorplanTaxonomy


def build_room_refine_case(batch: TrinityRoomBatch) -> RefineCase:
    batch_size, room_count = batch.token_mask.shape
    dtype = batch.z.dtype
    device = batch.z.device
    area_norm = batch.area_targets / batch.scale[:, None].square().clamp_min(1.0)
    token_mask = batch.token_mask.to(dtype)
    return RefineCase(
        area_norm=area_norm,
        token_mask=token_mask,
        mob_pos=token_mask,
        mob_shape=token_mask,
        cluster_id=batch.cluster_id,
        mib_id=torch.zeros_like(batch.cluster_id),
        boundary_code=batch.boundary_code,
        adjacency=batch.adjacency.to(dtype),
        pin_edge_xy_n=torch.zeros(batch_size, 1, 2, device=device, dtype=dtype),
        pin_edge_w=torch.zeros(batch_size, 1, device=device, dtype=dtype),
        pin_edge_block=torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=torch.long,
        ),
        anchor_z=torch.zeros(batch_size, room_count, 3, device=device, dtype=dtype),
        anchor_mask=torch.zeros(batch_size, room_count, 3, device=device, dtype=dtype),
        scale=batch.scale,
    )


@torch.no_grad()
def refine_room_latent(
    z: torch.Tensor,
    batch: TrinityRoomBatch,
    *,
    steps: int = 8,
    lr: float = 1.0e-3,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine normalized room latents and return ``(z, actual_xywh)``."""

    case = build_room_refine_case(batch)
    weights = weights or {
        "overlap": 10.0,
        "group": 1.0,
        "mib": 0.0,
        "boundary": 1.0,
        "wl": 1.0,
        "area": 1.0,
    }
    snapshots = refine_closed(
        z,
        case,
        weights,
        steps=int(steps),
        lr=float(lr),
        snapshots=(),
        compile_loop=False,
    )
    refined = snapshots[int(steps)]
    normalized_boxes = z_to_xywh(refined, case.area_norm, refined.new_ones(z.shape[0]))
    actual_boxes = normalized_boxes * batch.scale[:, None, None]
    return refined, actual_boxes


def legalize_room_boxes(
    boxes: torch.Tensor,
    batch: TrinityRoomBatch,
) -> torch.Tensor:
    """Run Trinity's overlap-free scale-pack legalizer per room batch row."""

    outputs = []
    for index in range(boxes.shape[0]):
        valid = batch.token_mask[index].detach().cpu().numpy().astype(bool)
        xywh = boxes[index].detach().cpu().numpy()[valid].astype(np.float64)
        areas = batch.area_targets[index].detach().cpu().numpy()[valid]
        adjacency = batch.adjacency[index].detach().cpu().numpy()[valid][:, valid]
        b2b_rows = np.argwhere(adjacency > 0)
        b2b = np.asarray(
            [
                (left, right, adjacency[left, right])
                for left, right in b2b_rows
                if left < right
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        constraints = np.zeros((len(xywh), 5), dtype=np.float64)
        constraints[:, 3] = batch.cluster_id[index].detach().cpu().numpy()[valid]
        constraints[:, 4] = batch.boundary_code[index].detach().cpu().numpy()[valid]
        instance = FloorplanInstance(
            block_count=len(xywh),
            area_targets=areas,
            constraints=constraints,
            b2b=b2b,
            p2b=np.zeros((0, 3), dtype=np.float64),
            pins_pos=np.zeros((0, 2), dtype=np.float64),
            target_positions=np.full((len(xywh), 4), -1.0, dtype=np.float64),
        )
        legalized = scale_pack_legalize(Placement(xywh, instance))
        row = torch.zeros_like(boxes[index])
        row[valid] = torch.from_numpy(legalized.xywh).to(row)
        outputs.append(row)
    return torch.stack(outputs)


def repair_room_raster(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    steps: int = 8,
    lr: float = 1.0e-3,
    max_legalizer_parts: int = 64,
) -> torch.Tensor:
    """Refine and legalize predicted room rectangles, then rerasterize."""

    batch = room_part_batch_from_labels(
        labels,
        taxonomy,
        footprint,
        max_rooms=32,
        max_parts=1024,
    )
    if not bool(batch.token_mask.any()):
        return labels
    _, refined_boxes = refine_room_latent(
        batch.z,
        batch,
        steps=steps,
        lr=lr,
    )
    if int(batch.token_mask.sum()) <= int(max_legalizer_parts):
        legalized = legalize_room_boxes(refined_boxes, batch)
    else:
        legalized = refined_boxes
    repaired = render_room_boxes(
        legalized,
        batch.room_classes,
        batch.token_mask,
        footprint,
        taxonomy,
    )
    expected = {
        int(value)
        for value in taxonomy.room_ids
        if bool((labels == int(value)).any())
    }
    present = {int(value) for value in repaired.unique().tolist()}
    if not expected.issubset(present):
        repaired = render_room_boxes(
            refined_boxes,
            batch.room_classes,
            batch.token_mask,
            footprint,
            taxonomy,
        )
        present = {int(value) for value in repaired.unique().tolist()}
    return repaired if expected.issubset(present) else labels


__all__ = [
    "build_room_refine_case",
    "legalize_room_boxes",
    "repair_room_raster",
    "refine_room_latent",
]

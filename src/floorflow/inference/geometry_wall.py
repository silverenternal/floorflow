"""Wall masks from geometry heatmap heads for branch-level fusion."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from floorflow.geometry.reconciliation import reconcile_plan
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan


@torch.no_grad()
def geometry_wall_mask(
    model,
    footprint: torch.Tensor,
    *,
    floor_ids: torch.Tensor | None = None,
    threshold: float = 0.7,
    dilate: int = 0,
) -> torch.Tensor | None:
    """Return a wall mask from a checkpoint's geometry branch when available."""

    if footprint.ndim == 3:
        footprint_bchw = footprint[:, None]
    else:
        footprint_bchw = footprint
    state = torch.zeros(
        footprint_bchw.shape[0],
        model.target_channels,
        *footprint_bchw.shape[-2:],
        device=footprint_bchw.device,
        dtype=footprint_bchw.dtype,
    )
    _velocity, auxiliary = model.forward_with_aux(
        state,
        torch.zeros(
            footprint_bchw.shape[0],
            device=footprint_bchw.device,
            dtype=footprint_bchw.dtype,
        ),
        footprint_bchw,
        floor_ids=floor_ids,
    )
    geometry = auxiliary.get("geometry")
    if not isinstance(geometry, dict):
        return None
    logits = geometry.get("wall_centerline_logits")
    if logits is None:
        return None
    wall = (logits.sigmoid() >= float(threshold)).long()
    radius = max(0, int(dilate))
    if radius > 0:
        wall = F.max_pool2d(
            wall.float(),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        ).long()
    return wall[:, 0].bool() & (footprint_bchw[:, 0] > 0.5)


def apply_wall_face_rooms(
    labels: torch.Tensor,
    wall_mask: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
    *,
    reconcile: bool = False,
    reconcile_use_ip: bool = False,
) -> torch.Tensor:
    """Replace room labels by wall-graph faces when a planar face readout exists."""

    output = labels.clone()
    labels_np = labels.detach().cpu().numpy()
    walls_np = wall_mask.detach().cpu().numpy()
    footprint_np = footprint.detach().cpu().numpy() > 0
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    for batch_index in range(labels_np.shape[0]):
        dummy = labels_np[batch_index].copy()
        dummy[walls_np[batch_index]] = taxonomy.class_id("wall")
        plan = raster_labels_to_wall_plan(dummy, taxonomy)
        if reconcile:
            plan = reconcile_plan(
                plan,
                dummy,
                taxonomy,
                use_ip=reconcile_use_ip,
            )
        if not plan.rooms:
            continue
        face = np.zeros_like(dummy, dtype=np.int32)
        for room in plan.rooms:
            name = str(room.get("label", "undefined"))
            try:
                class_id = taxonomy.class_id(name)
            except ValueError:
                class_id = taxonomy.class_id("room:Undefined")
            points = np.rint(np.asarray(room["polygon"], dtype=np.float64)).astype(np.int32)
            cv2.fillPoly(face, [points], int(class_id))
        valid = np.isin(face, room_ids) & footprint_np[batch_index]
        output[batch_index][valid] = torch.from_numpy(face[valid].astype(np.int64)).to(
            device=labels.device,
            dtype=labels.dtype,
        )
    return output


__all__ = ["apply_wall_face_rooms", "geometry_wall_mask"]

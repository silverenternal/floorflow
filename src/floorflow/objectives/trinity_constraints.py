"""Room-layout semantics for Trinity's differentiable constraint terms."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

import trinity  # noqa: F401  (populates Trinity registries)
from trinity.decode import z_to_xywh
from trinity.losses.base import LossContext
from trinity.registry import LOSS, build

from floorflow.representation.trinity_latent import TrinityRoomBatch


@dataclass(frozen=True)
class TrinityConstraintWeights:
    overlap: float = 0.05
    group: float = 0.02
    mib: float = 0.0
    boundary: float = 0.02
    wl: float = 0.01
    area: float = 0.02


def build_room_loss_context(
    z: torch.Tensor,
    batch: TrinityRoomBatch,
    *,
    time: torch.Tensor | None = None,
) -> LossContext:
    if z.shape != batch.z.shape:
        raise ValueError("z must match TrinityRoomBatch shape")
    xywh = z_to_xywh(z, batch.area_targets, batch.scale)
    batch_size, room_count = batch.token_mask.shape
    if time is None:
        time = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
    pin_xy = torch.zeros(batch_size, 1, 2, device=z.device, dtype=z.dtype)
    pin_weight = torch.zeros(batch_size, 1, device=z.device, dtype=z.dtype)
    pin_block = torch.zeros(
        batch_size,
        1,
        device=z.device,
        dtype=torch.long,
    )
    return LossContext(
        pred=z,
        target=batch.z,
        weight=torch.ones(batch_size, device=z.device, dtype=z.dtype),
        t=time,
        anchor_mask=torch.zeros(
            batch_size,
            room_count,
            3,
            device=z.device,
            dtype=z.dtype,
        ),
        xywh=xywh,
        xywh_gt=batch.xywh,
        area_targets=batch.area_targets,
        scale=batch.scale,
        cluster_id=batch.cluster_id,
        boundary_code=batch.boundary_code,
        mib_id=torch.zeros_like(batch.cluster_id),
        token_mask=batch.token_mask.to(z.dtype),
        pin_edge_xy_n=pin_xy,
        pin_edge_w=pin_weight,
        pin_edge_block=pin_block,
        mob_pos=batch.token_mask.to(z.dtype),
        mob_shape=batch.token_mask.to(z.dtype),
        adjacency=batch.adjacency,
    )


def subset_room_batch(
    batch: TrinityRoomBatch,
    indices: torch.Tensor,
) -> TrinityRoomBatch:
    return replace(
        batch,
        z=batch.z[indices],
        area_targets=batch.area_targets[indices],
        scale=batch.scale[indices],
        xywh=batch.xywh[indices],
        token_mask=batch.token_mask[indices],
        room_classes=batch.room_classes[indices],
        cluster_id=batch.cluster_id[indices],
        boundary_code=batch.boundary_code[indices],
        adjacency=batch.adjacency[indices][:, indices],
        room_uid=(
            batch.room_uid[indices] if batch.room_uid is not None else None
        ),
    )


def trinity_constraint_loss(
    z: torch.Tensor,
    batch: TrinityRoomBatch,
    *,
    weights: TrinityConstraintWeights | None = None,
    time: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Adapt Trinity's six floorplanning physics terms to room geometry."""

    weights = weights or TrinityConstraintWeights()
    valid_rows = batch.token_mask.any(dim=1)
    if not bool(valid_rows.any()):
        zero = z.sum() * 0.0
        return zero, {
            "overlap": zero.detach(),
            "group": zero.detach(),
            "mib": zero.detach(),
            "boundary": zero.detach(),
            "wl": zero.detach(),
            "area": zero.detach(),
            "trinity_total": zero.detach(),
        }
    if not bool(valid_rows.all()):
        indices = torch.nonzero(valid_rows, as_tuple=False).flatten()
        z = z[indices]
        batch = subset_room_batch(batch, indices)
        if time is not None:
            time = time[indices]
    context = build_room_loss_context(z, batch, time=time)
    total = z.sum() * 0.0
    components = {}
    for name in ("overlap", "group", "mib", "boundary", "wl", "area"):
        weight = float(getattr(weights, name))
        if weight <= 0:
            components[name] = z.sum().detach() * 0.0
            continue
        term = build({"name": name, "weight": weight}, LOSS)
        loss, diagnostics = term(context)
        total = total + loss
        components[name] = diagnostics[name]
    components["trinity_total"] = total.detach()
    return total, components


__all__ = [
    "TrinityConstraintWeights",
    "build_room_loss_context",
    "subset_room_batch",
    "trinity_constraint_loss",
]

from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.objectives.trinity_constraints import (
    TrinityConstraintWeights,
    trinity_constraint_loss,
)
from floorflow.representation.trinity_latent import room_batch_from_labels


def _batch():
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[:, :16] = taxonomy.room_ids[0]
    labels[:, 16:] = taxonomy.room_ids[1]
    footprint = torch.ones(1, 1, 32, 32)
    return room_batch_from_labels(
        labels,
        taxonomy,
        footprint,
        max_rooms=4,
    )


def test_trinity_room_constraints_are_finite_and_differentiable() -> None:
    batch = _batch()
    z = (batch.z + 0.01 * torch.randn_like(batch.z)).requires_grad_(True)
    weights = TrinityConstraintWeights(
        overlap=0.05,
        group=0.02,
        mib=0.0,
        boundary=0.02,
        wl=0.01,
        area=0.02,
    )
    loss, components = trinity_constraint_loss(
        z,
        batch,
        weights=weights,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert set(components) == {
        "overlap",
        "group",
        "mib",
        "boundary",
        "wl",
        "area",
        "trinity_total",
    }


def test_trinity_constraint_does_not_require_target_control() -> None:
    batch = _batch()
    loss, components = trinity_constraint_loss(batch.z, batch)
    assert torch.isfinite(loss)
    assert components["trinity_total"] >= 0


def test_empty_room_rows_do_not_explode_physics() -> None:
    batch = _batch()
    empty = type(batch)(
        z=torch.zeros_like(batch.z),
        area_targets=torch.zeros_like(batch.area_targets),
        scale=torch.zeros_like(batch.scale),
        xywh=torch.zeros_like(batch.xywh),
        token_mask=torch.zeros_like(batch.token_mask),
        room_classes=torch.zeros_like(batch.room_classes),
        cluster_id=torch.zeros_like(batch.cluster_id),
        boundary_code=torch.zeros_like(batch.boundary_code),
        adjacency=torch.zeros_like(batch.adjacency),
        room_uid=torch.zeros_like(batch.room_uid),
    )
    loss, components = trinity_constraint_loss(empty.z, empty)
    assert loss.item() == 0.0
    assert components["trinity_total"].item() == 0.0

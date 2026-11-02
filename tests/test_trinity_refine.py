from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.trinity_refine import refine_room_latent
from floorflow.inference.trinity_refine import legalize_room_boxes
from floorflow.representation.trinity_latent import room_batch_from_labels


def test_trinity_closed_form_refiner_runs_on_room_latents() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[:, :16] = taxonomy.room_ids[0]
    labels[:, 16:] = taxonomy.room_ids[1]
    footprint = torch.ones(1, 1, 32, 32)
    batch = room_batch_from_labels(labels, taxonomy, footprint, max_rooms=4)
    initial = batch.z + 0.05 * torch.randn_like(batch.z)
    refined, boxes = refine_room_latent(
        initial,
        batch,
        steps=3,
        lr=1.0e-4,
    )
    assert refined.shape == batch.z.shape
    assert boxes.shape == batch.xywh.shape
    assert torch.isfinite(refined).all()
    assert torch.isfinite(boxes).all()


def test_trinity_legalizer_returns_nonoverlapping_boxes() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[:, :16] = taxonomy.room_ids[0]
    labels[:, 16:] = taxonomy.room_ids[1]
    batch = room_batch_from_labels(
        labels,
        taxonomy,
        torch.ones(1, 1, 32, 32),
        max_rooms=4,
    )
    boxes = batch.xywh.clone()
    boxes[0, 1, :2] = boxes[0, 0, :2] + 4
    legalized = legalize_room_boxes(boxes, batch)
    valid = batch.token_mask[0]
    left = legalized[0, valid]
    for index in range(len(left)):
        for other in range(index + 1, len(left)):
            x0, y0, w0, h0 = left[index]
            x1, y1, w1, h1 = left[other]
            overlap_x = max(0.0, min(x0 + w0, x1 + w1) - max(x0, x1))
            overlap_y = max(0.0, min(y0 + h0, y1 + h1) - max(y0, y1))
            assert overlap_x * overlap_y < 1.0e-6

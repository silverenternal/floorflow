from __future__ import annotations

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.trinity_latent import (
    decode_room_boxes,
    render_room_boxes,
    room_batch_from_labels,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )


def test_cubicasa_room_batch_round_trips_inside_trinity_latent() -> None:
    taxonomy = _taxonomy()
    rooms = taxonomy.room_ids[:2]
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[:, :16] = rooms[0]
    labels[:, 16:] = rooms[1]
    footprint = torch.ones(1, 1, 32, 32)
    batch = room_batch_from_labels(
        labels,
        taxonomy,
        footprint,
        max_rooms=4,
    )
    valid = batch.token_mask[0].numpy()
    decoded = decode_room_boxes(
        batch.z[0].numpy()[valid],
        batch.area_targets[0].numpy()[valid],
        float(batch.scale[0]),
    )
    assert np.allclose(decoded, batch.xywh[0].numpy()[valid], atol=1e-4)
    assert batch.adjacency.shape == (1, 4, 4)
    assert int((batch.adjacency[0] > 0).sum()) >= 2
    labels = render_room_boxes(
        batch.xywh,
        batch.room_classes,
        batch.token_mask,
        footprint,
        taxonomy,
    )
    assert labels.shape == (1, 32, 32)
    assert int((labels == rooms[0]).sum()) > 0
    assert int((labels == rooms[1]).sum()) > 0


def test_boundary_codes_and_padding_are_explicit() -> None:
    taxonomy = _taxonomy()
    room = taxonomy.room_ids[0]
    labels = torch.zeros(1, 20, 20, dtype=torch.long)
    labels[:, 4:16, 4:16] = room
    footprint = torch.ones(1, 1, 20, 20)
    batch = room_batch_from_labels(
        labels,
        taxonomy,
        footprint,
        max_rooms=3,
    )
    assert int(batch.boundary_code[0, 0]) == 0
    assert not bool(batch.token_mask[0, 1:].any())
    assert torch.equal(
        batch.constraints[0, 1:],
        torch.zeros_like(batch.constraints[0, 1:]),
    )

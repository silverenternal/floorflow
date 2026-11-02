from __future__ import annotations

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.trinity_polygons import (
    decompose_mask_to_rectangles,
    room_part_batch_from_labels,
)


def _rasterize(rectangles, shape):
    result = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in rectangles:
        result[y0:y1, x0:x1] = True
    return result


def test_rectilinear_decomposition_is_lossless_on_concave_masks() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[1:11, 1:5] = True
    mask[7:11, 5:10] = True
    mask[5:7, 5:8] = True
    rectangles = decompose_mask_to_rectangles(mask)
    assert np.array_equal(_rasterize(rectangles, mask.shape), mask)
    assert len(rectangles) < int(mask.sum())


def test_part_batch_preserves_room_identity_and_exact_cover() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, 1:12, 1:5] = taxonomy.room_ids[0]
    labels[:, 7:12, 5:12] = taxonomy.room_ids[0]
    labels[:, 1:7, 12:15] = taxonomy.room_ids[1]
    batch = room_part_batch_from_labels(
        labels,
        taxonomy,
        torch.ones(1, 1, 16, 16),
        max_rooms=4,
        max_parts=16,
    )
    valid = batch.token_mask[0]
    assert batch.room_uid is not None
    assert len(batch.room_uid[0, valid].unique()) == 2
    painted = torch.zeros(1, 16, 16, dtype=torch.long)
    room_ids = torch.tensor(taxonomy.room_ids)
    for box, class_index in zip(
        batch.xywh[0, valid].long(),
        batch.room_classes[0, valid],
    ):
        x0, y0, width, height = box.tolist()
        painted[0, y0 : y0 + height, x0 : x0 + width] = room_ids[class_index]
    assert torch.equal(painted, labels)

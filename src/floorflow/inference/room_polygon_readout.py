"""Deterministic room-mask to vector readout variants."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.core import compose_known_exterior
from floorflow.geometry.room_polygons import room_polygon_rows
from floorflow.heads.structured_objects import partition_wall_mask


def _rasterize_rows(
    rows: list[dict],
    shape: tuple[int, int],
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    label_lookup = {
        taxonomy.class_names[index]: index
        for index in range(taxonomy.num_classes)
    }
    output = np.zeros(shape, dtype=np.int32)
    ranked = sorted(
        rows,
        key=lambda row: abs(
            float(
                cv2.contourArea(
                    np.asarray(row["polygon"], dtype=np.float32)
                )
            )
        ),
        reverse=True,
    )
    for row in ranked:
        points = np.rint(
            np.asarray(row["polygon"], dtype=np.float64)
        ).astype(np.int32)
        if len(points) < 3:
            continue
        cv2.fillPoly(
            output,
            [points],
            int(label_lookup[row["label"]]),
        )
    return output.astype(np.int64)


def vectorize_room_masks(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    epsilon: float,
    minimum_area: int = 12,
) -> np.ndarray:
    """Convert room masks to simplified polygons and rebuild wall seams."""

    source = np.asarray(labels, dtype=np.int64)
    area = np.asarray(footprint, dtype=bool)
    rows = room_polygon_rows(
        source,
        taxonomy,
        min_area=minimum_area,
        epsilon=float(epsilon),
        footprint=area,
        tile_footprint=True,
    )
    rooms = _rasterize_rows(rows, source.shape, taxonomy) * area
    room_tensor = torch.from_numpy(rooms)[None]
    seam = partition_wall_mask(room_tensor, radius=1) & torch.from_numpy(
        area
    )[None]
    opening_ids = np.asarray(taxonomy.opening_ids, dtype=np.int64)
    opening = np.isin(source, opening_ids)
    support = F.max_pool2d(
        seam.float()[:, None],
        5,
        stride=1,
        padding=2,
    )[0, 0] > 0.5
    opening &= support.numpy()
    output = rooms.copy()
    output[seam[0].numpy()] = taxonomy.class_id("wall")
    output[opening] = source[opening]
    return compose_known_exterior(
        torch.from_numpy(output)[None],
        torch.from_numpy(area.astype(np.float32))[None],
        opening_ids=taxonomy.opening_ids,
    )[0].numpy()


__all__ = ["vectorize_room_masks"]

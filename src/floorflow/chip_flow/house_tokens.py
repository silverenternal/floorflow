"""Convert CubiCasa room instances to domain-aligned rectangular tokens."""

from __future__ import annotations

import math

import numpy as np
import torch

from floorflow.chip_flow.proposals import prepare_flow_pair
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.core import extract_room_instances


def house_feature_matrix(
    boxes: torch.Tensor,
    class_ids: torch.Tensor,
    footprint: np.ndarray,
) -> torch.Tensor:
    """Return ten box/program features with the chip model's input width."""

    areas = boxes[:, 2] * boxes[:, 3]
    area_total = areas.sum().clamp_min(1.0e-6)
    centers = boxes[:, :2] + boxes[:, 2:] * 0.5
    image_area = float(footprint.shape[0] * footprint.shape[1])
    aspect = torch.log(boxes[:, 2].clamp_min(1.0e-6) / boxes[:, 3].clamp_min(1.0e-6))
    angle = class_ids.to(areas.dtype) * (2.0 * math.pi / 31.0)
    fp_y, fp_x = np.nonzero(footprint)
    if fp_x.size:
        fp_width = float(fp_x.max() - fp_x.min() + 1) / footprint.shape[1]
        fp_height = float(fp_y.max() - fp_y.min() + 1) / footprint.shape[0]
    else:
        fp_width = fp_height = 1.0
    constants = torch.tensor([fp_width, fp_height], dtype=areas.dtype)
    return torch.stack(
        [
            torch.log(areas / (area_total / len(areas))),
            centers[:, 0],
            centers[:, 1],
            aspect,
            areas / image_area,
            torch.sin(angle),
            torch.cos(angle),
            boxes[:, 2],
            boxes[:, 3],
            constants[0].expand(len(areas)),
        ],
        dim=1,
    )


def room_box_case(
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    max_rooms: int = 32,
    min_area: int = 8,
    source: str | None = None,
) -> dict | None:
    """Build one normalized room-bounding-box case."""

    instances = sorted(
        extract_room_instances(target, taxonomy, min_area=min_area),
        key=lambda item: item.area,
        reverse=True,
    )[:max_rooms]
    if not instances:
        return None
    height, width = target.shape
    xyxy = torch.tensor([item.bbox for item in instances], dtype=torch.float32)
    boxes = torch.stack(
        [
            xyxy[:, 0] / width,
            xyxy[:, 1] / height,
            (xyxy[:, 2] - xyxy[:, 0]) / width,
            (xyxy[:, 3] - xyxy[:, 1]) / height,
        ],
        dim=1,
    ).clamp_min(1.0e-4)
    class_ids = torch.tensor([item.class_id for item in instances], dtype=torch.long)
    features = house_feature_matrix(boxes, class_ids, footprint)
    areas = boxes[:, 2] * boxes[:, 3]
    constraints = torch.zeros(len(instances), 5)
    ys, xs = np.nonzero(footprint)
    if xs.size:
        left, top = float(xs.min()), float(ys.min())
        right, bottom = float(xs.max() + 1), float(ys.max() + 1)
        edge = 2.0
        constraints[:, 4] = (
            (xyxy[:, 0] <= left + edge).float()
            + (xyxy[:, 2] >= right - edge).float() * 2
            + (xyxy[:, 1] <= top + edge).float() * 4
            + (xyxy[:, 3] >= bottom - edge).float() * 8
        )
    edges = []
    for left_index in range(len(instances)):
        for right_index in range(left_index + 1, len(instances)):
            gap_x = max(
                float(xyxy[left_index, 0] - xyxy[right_index, 2]),
                float(xyxy[right_index, 0] - xyxy[left_index, 2]),
                0.0,
            )
            gap_y = max(
                float(xyxy[left_index, 1] - xyxy[right_index, 3]),
                float(xyxy[right_index, 1] - xyxy[left_index, 3]),
                0.0,
            )
            if min(gap_x, gap_y) <= 2.0:
                edges.append([left_index, right_index, 1.0])
    b2b = torch.tensor(edges, dtype=torch.float32)
    if not b2b.numel():
        b2b = torch.zeros(0, 3)
    source_latent, target_latent = prepare_flow_pair(boxes, areas)
    return {
        "areas": areas,
        "boxes": boxes,
        "b2b": b2b,
        "p2b": torch.zeros(0, 3),
        "pins": torch.zeros(0, 2),
        "constraints": constraints,
        "features": features,
        "source_latent": source_latent,
        "target_latent": target_latent,
        "source": source,
        "source_index": 0,
    }


__all__ = ["house_feature_matrix", "room_box_case"]

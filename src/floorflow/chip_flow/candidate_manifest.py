"""Build candidate-consensus manifests from saved chip case artifacts."""

from __future__ import annotations

import math

import numpy as np
import torch

from floorflow.chip_flow.batched import (
    batch_grouping_proxy,
    batch_physics_terms,
)
from floorflow.chip_flow.model import encode_latents

SOURCE_GEOMETRY_KEYS = (
    "gt_boxes",
    "areas",
    "constraints",
    "b2b",
    "p2b",
    "pins",
)

TARGET_FREE_FEATURE_NAMES = (
    "log_block_count",
    "b2b_edges_per_block",
    "p2b_edges_per_block",
    "b2b_hpwl",
    "p2b_hpwl",
    "outline_area",
    "log_outline_aspect",
    "legalizer_rung",
    "legalizer_solves",
    "legalizer_reshaped",
    "legalization_displacement",
    "shape_displacement",
    "grouping_proxy",
    "boundary_proxy",
)


def _target_free_features(row: dict, boxes: np.ndarray) -> list[float]:
    areas = np.asarray(row["areas"], dtype=np.float64)
    count = len(areas)
    scale = math.sqrt(max(float(areas.sum()), 1.0e-9))
    centers = boxes[:, :2] + boxes[:, 2:] / 2.0
    b2b = np.asarray(row["b2b"], dtype=np.float64).reshape(-1, 3)
    p2b = np.asarray(row["p2b"], dtype=np.float64).reshape(-1, 3)
    if len(b2b):
        left = b2b[:, 0].astype(int)
        right = b2b[:, 1].astype(int)
        weights = b2b[:, 2]
        b2b_hpwl = float(
            (
                np.abs(centers[left] - centers[right]).sum(axis=1) * weights
            ).sum()
            / (scale * max(float(weights.sum()), 1.0e-9))
        )
    else:
        b2b_hpwl = 0.0
    if len(p2b):
        pins = np.asarray(row["pins"], dtype=np.float64)
        pin_id = p2b[:, 0].astype(int)
        block_id = p2b[:, 1].astype(int)
        weights = p2b[:, 2]
        p2b_hpwl = float(
            (
                np.abs(pins[pin_id] - centers[block_id]).sum(axis=1)
                * weights
            ).sum()
            / (scale * max(float(weights.sum()), 1.0e-9))
        )
    else:
        p2b_hpwl = 0.0
    x0, y0 = boxes[:, :2].min(axis=0)
    x1, y1 = (boxes[:, :2] + boxes[:, 2:]).max(axis=0)
    width = max(float(x1 - x0), 1.0e-9)
    height = max(float(y1 - y0), 1.0e-9)
    raw_boxes = np.asarray(row.get("raw_xywh", boxes), dtype=np.float64)
    raw_centers = raw_boxes[:, :2] + raw_boxes[:, 2:] / 2.0
    displacement = float(
        np.linalg.norm(raw_centers - centers, axis=1).mean() / scale
    )
    shape_displacement = float(
        np.abs(raw_boxes[:, 2:] - boxes[:, 2:]).mean()
        / math.sqrt(max(float(areas.mean()), 1.0e-9))
    )
    constraints = torch.as_tensor(row["constraints"], dtype=torch.float32)[None]
    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)[None]
    physics = batch_physics_terms(
        boxes_tensor,
        torch.as_tensor(areas, dtype=torch.float32)[None],
        constraints,
        torch.as_tensor(b2b, dtype=torch.float32)[None],
        torch.as_tensor(p2b, dtype=torch.float32)[None],
        torch.as_tensor(row["pins"], dtype=torch.float32)[None],
        torch.tensor([len(b2b)]),
        torch.tensor([len(p2b)]),
        torch.tensor([len(row["pins"])]),
    )
    grouping = float(batch_grouping_proxy(boxes_tensor, constraints)[0]) / scale
    boundary = float(physics["boundary"][0])
    return [
        math.log1p(count),
        float(row.get("n_b2b", len(b2b))) / max(count, 1),
        float(row.get("n_p2b", len(p2b))) / max(count, 1),
        b2b_hpwl,
        p2b_hpwl,
        width * height / scale**2,
        math.log(width / height),
        float(row.get("legalizer_rung", 0.0)) / 5.0,
        float(row.get("legalizer_solves", 0.0)) / 20.0,
        float(row.get("legalizer_reshaped", 0.0)),
        displacement,
        shape_displacement,
        grouping,
        boundary,
    ]


def candidate_manifest_from_sources(
    sources: dict[str, list[dict]],
    *,
    reference_source: str | None = None,
) -> list[dict]:
    """Merge matched source case rows into one candidate list per case."""

    if not sources:
        raise ValueError("candidate manifest requires at least one source")
    if reference_source is not None and reference_source not in sources:
        raise ValueError("reference source is not present in candidate sources")
    source_order = (
        [reference_source, *sorted(set(sources) - {reference_source})]
        if reference_source is not None
        else sorted(sources)
    )
    by_index: dict[int, list[tuple[str, dict]]] = {}
    layouts: dict[int, dict] = {}
    for source_name in source_order:
        rows = sources[source_name]
        for row in rows:
            index = int(row["index"])
            by_index.setdefault(index, []).append((source_name, row))
            layouts.setdefault(index, row)
    cases = []
    for index in sorted(by_index):
        base = layouts[index]
        expected = {
            key: np.asarray(base[key])
            for key in ("gt_boxes", "areas", "b2b", "p2b", "pins")
            if key in base
        }
        for source_name, row in by_index[index]:
            for key, value in expected.items():
                if key not in row or not np.allclose(
                    value,
                    np.asarray(row[key]),
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"source {source_name} case {index} mismatches {key}"
                    )
        target_boxes = torch.as_tensor(
            np.asarray(base["gt_boxes"], dtype=np.float32)
        )[None]
        areas = torch.as_tensor(np.asarray(base["areas"], dtype=np.float32))[None]
        target_state = encode_latents(target_boxes, areas)[0].numpy()
        source_geometry = {
            key: np.asarray(base[key]).tolist()
            for key in SOURCE_GEOMETRY_KEYS
            if key in base
        }
        candidates = []
        for source_name, row in by_index[index]:
            boxes_value = row.get("legalized_xywh", row.get("raw_xywh"))
            if boxes_value is None:
                raise ValueError(
                    f"source {source_name} case {index} lacks raw/legalized boxes"
                )
            boxes = np.asarray(boxes_value, dtype=np.float64)
            state_tensor = torch.as_tensor(boxes.astype(np.float32))[None]
            state = encode_latents(state_tensor, areas)[0].numpy()
            quality = (
                row.get("legalized_hard_cost")
                if row.get("legalized_feasible", False)
                else row.get("raw_hard_cost", 10.0)
            )
            candidates.append(
                {
                    "source": source_name,
                    "state": state.tolist(),
                    "features": _target_free_features(row, boxes),
                    "quality": float(quality),
                    "valid": True,
                }
            )
        cases.append(
            {
                "index": index,
                "target_state": target_state.tolist(),
                "source_geometry": source_geometry,
                "candidates": candidates,
            }
        )
    return cases


__all__ = [
    "SOURCE_GEOMETRY_KEYS",
    "TARGET_FREE_FEATURE_NAMES",
    "candidate_manifest_from_sources",
]

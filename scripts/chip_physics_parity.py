#!/usr/bin/env python3
"""Parity check: our chip physics terms vs the official FloorSet checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from shapely.geometry import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party/floorset-src"))

from lite_dataset_test import FloorplanDatasetLiteTest
from utils import (
    check_boundary_const,
    check_clust_const,
    check_mib_const,
)

from floorflow.chip_physics import chip_physics_loss


def polygons_to_boxes(polys: torch.Tensor) -> torch.Tensor:
    boxes = []
    for poly in polys:
        valid = poly[(poly >= -0.5).all(dim=1)]
        low = valid.min(dim=0).values
        high = valid.max(dim=0).values
        boxes.append(torch.stack([low[0], low[1], high[0] - low[0], high[1] - low[1]]))
    return torch.stack(boxes)


def main() -> None:
    dataset = FloorplanDatasetLiteTest(str(REPO_ROOT / "third_party/floorset-src"))
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        areas, b2b, p2b, pins, constraints = item["input"]
        polygons = item["label"][0]
        boxes = polygons_to_boxes(polygons)
        terms = chip_physics_loss(boxes, areas, constraints, b2b, p2b, pins)
        solution = [
            Polygon([point for point in poly.tolist() if point != [-1.0, -1.0]])
            for poly in polygons
        ]
        width = max(poly.bounds[2] for poly in solution)
        height = max(poly.bounds[3] for poly in solution)
        official_boundary = check_boundary_const(
            constraints[:, 4], solution, solution, width, height
        )
        official_mib = check_mib_const(constraints[:, 2], solution, solution)
        official_cluster = check_clust_const(constraints[:, 3], solution, solution)
        rows.append(
            {
                "case": index,
                "blocks": int(areas.numel()),
                "overlap": float(terms["overlap"]),
                "group": float(terms["group"]),
                "mib": float(terms["mib"]),
                "boundary": float(terms["boundary"]),
                "official_boundary": int(official_boundary),
                "official_mib": int(official_mib),
                "official_cluster": int(official_cluster),
            }
        )
    positive = {
        "boundary": sum(row["boundary"] > 1.0e-6 for row in rows),
        "mib": sum(row["mib"] > 1.0e-6 for row in rows),
        "group": sum(row["group"] > 1.0e-6 for row in rows),
        "overlap": sum(row["overlap"] > 1.0e-6 for row in rows),
    }
    official = {
        "boundary": sum(row["official_boundary"] > 0 for row in rows),
        "mib": sum(row["official_mib"] > 0 for row in rows),
        "group": sum(row["official_cluster"] > 0 for row in rows),
    }
    report = {
        "cases": len(rows),
        "positive_cases": positive,
        "official_positive_cases": official,
        "agreement": {
            "boundary": sum(
                (row["boundary"] > 1.0e-6) == (row["official_boundary"] > 0)
                for row in rows
            )
            / len(rows),
            "mib": sum(
                (row["mib"] > 1.0e-6) == (row["official_mib"] > 0)
                for row in rows
            )
            / len(rows),
            "group": sum(
                (row["group"] > 1.0e-6) == (row["official_cluster"] > 0)
                for row in rows
            )
            / len(rows),
        },
        "rows": rows,
    }
    output = REPO_ROOT / "outputs/reports/chip_physics_parity_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cases", "positive_cases", "official_positive_cases", "agreement")}, indent=2))


if __name__ == "__main__":
    main()

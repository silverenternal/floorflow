#!/usr/bin/env python3
"""Evaluate external predicted-control rasters with the frozen structural suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate_raster_control_router_screen import _metric_row

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.coarse_structure_refiner import coarse_structure_control

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/external_control_structural_screen_v1.json",
    )
    parser.add_argument("--method", default="external")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    paths = sorted(args.prediction_root.glob("*.npz"))[
        int(args.offset) : int(args.offset) + int(args.limit)
    ]
    rows = []
    room_count = len(taxonomy.room_ids)
    for path in paths:
        with np.load(path) as payload:
            prediction = payload["prediction"].astype(np.int64)
            target = payload["target"].astype(np.int64)
            footprint = payload["footprint"].astype(np.float32)
        target_tensor = torch.from_numpy(target)
        target_clean = coarse_structure_control(
            target_tensor[None],
            torch.from_numpy(footprint)[None],
            taxonomy,
            size=256,
        )[0]
        rows.append(
            {
                "case": path.stem,
                **_metric_row(
                    torch.from_numpy(prediction),
                    target_tensor,
                    torch.from_numpy(footprint),
                    taxonomy,
                    target_clean,
                    room_count,
                ),
            }
        )
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key != "case"
    }
    report = {
        "schema_version": "external-control-structural-screen-v1",
        "method": args.method,
        "prediction_root": str(args.prediction_root),
        "validation_cases": len(rows),
        "validation_offset": args.offset,
        "test_opened": False,
        "summary": summary,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

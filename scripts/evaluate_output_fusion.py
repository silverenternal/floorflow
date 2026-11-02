#!/usr/bin/env python3
"""Evaluate deterministic room-centric + wall-first output fusion."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from floorflow.evaluation.fpeval_bridge import summarise_fpeval_report, to_fpeval_plan
from floorflow.geometry.geometry_graph import dense_room_plan
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rasters", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1/taxonomy.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _plan(labels: np.ndarray, wall_mask: np.ndarray, taxonomy: FloorplanTaxonomy):
    combined = labels.copy()
    combined[wall_mask] = taxonomy.class_id("wall")
    return raster_labels_to_wall_plan(combined, taxonomy)


def _process(path: str, taxonomy_path: str) -> dict:
    taxonomy = FloorplanTaxonomy.read(taxonomy_path)
    with np.load(path) as payload:
        prediction = payload["prediction"]
        target = payload["target"]
        room_support = payload["room_support"]
        footprint = payload["footprint"]
    wall_mask = prediction == taxonomy.class_id("wall")
    opening_mask = np.isin(prediction, taxonomy.opening_ids)
    boundary_mask = partition_wall_mask(
        torch.from_numpy(room_support)[None]
    )[0].numpy()
    wall_plan = _plan(room_support, wall_mask | opening_mask, taxonomy)
    room_base = _plan(room_support, boundary_mask | opening_mask, taxonomy)
    room_plan = dense_room_plan(
        room_support,
        footprint,
        room_base,
        taxonomy,
    )
    # Contour-only inference has no independent interior ink channel.  A room
    # boundary would therefore be a self-referential donor gate, which the
    # frozen audit shows harms wall validity.  Abstain and emit wall-first.
    fused_plan = wall_plan
    target_plan = to_fpeval_plan(raster_labels_to_wall_plan(target, taxonomy))
    from fpeval.metrics import evaluate

    summaries = {}
    for name, plan in (
        ("room_centric", room_plan),
        ("wall_first", wall_plan),
        ("fused", fused_plan),
    ):
        report = evaluate(to_fpeval_plan(plan), target_plan, tol_frac=0.05)
        summaries[name] = summarise_fpeval_report(report)
    return {"case": Path(path).stem, **summaries}


def _metric_summary(rows: list[dict], name: str, key: str) -> dict[str, float]:
    return {
        "mean": float(np.mean([row[name][key] for row in rows])),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.rasters.glob("*.npz"))
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(_process, str(path), str(args.taxonomy))
            for path in paths
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 100 == 0 or completed == len(paths):
                print(f"evaluated {completed}/{len(paths)}", flush=True)
    keys = (
        "room_f1",
        "opening_f1",
        "wall_f1",
        "validity",
        "edit_cost",
    )
    metrics = {
        name: {
            key: _metric_summary(rows, name, key)["mean"]
            for key in keys
        }
        for name in ("room_centric", "wall_first", "fused")
    }
    checks = {}
    for baseline in ("room_centric", "wall_first"):
        checks[f"fused_vs_{baseline}"] = {
            key: metrics["fused"][key] - metrics[baseline][key]
            for key in keys
        }
    report = {
        "case_count": len(rows),
        "best_of_n_used": False,
        "fusion_policy": "wall_first_abstain_without_interior_ink",
        "metrics": metrics,
        "deltas": checks,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "deltas": checks}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Precompute wall-face masks, adjacency, and room-type targets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from floorflow.heads.face_semantics import build_face_targets
from floorflow.data.geometry_cache import load_geometry_cache
from floorflow.geometry.geometry_graph import wall_plan_from_cache
from floorflow.data.taxonomy import FloorplanTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_geometry_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_faces_v1",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/reports/face_targets_v1.json",
    )
    return parser.parse_args()


def _process(
    source: Path,
    geometry_root: Path,
    output_root: Path,
    taxonomy: FloorplanTaxonomy,
    overwrite: bool,
) -> dict:
    relative = source.relative_to(geometry_root)
    target = output_root / relative.with_name(
        relative.name.replace(".geometry.npz", ".faces.npz")
    )
    if target.is_file() and not overwrite:
        return {"case": str(relative), "status": "reused"}
    try:
        geometry = load_geometry_cache(source)
        plan = wall_plan_from_cache(source, taxonomy)
        faces = build_face_targets(plan, geometry["target"], taxonomy)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".tmp-{os.getpid()}.npz")
        np.savez_compressed(
            temporary,
            masks=(
                np.stack(faces.masks)
                if faces.masks
                else np.zeros((0, *geometry["target"].shape), dtype=bool)
            ),
            adjacency=faces.adjacency,
            classes=faces.classes,
            purity=faces.purity,
        )
        os.replace(temporary, target)
        return {
            "case": str(relative),
            "status": "written",
            "faces": len(faces.masks),
        }
    except Exception as error:
        return {
            "case": str(relative),
            "status": "error",
            "error": repr(error),
        }


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.geometry_root / "taxonomy.json")
    rows = []
    for split in args.splits:
        sources = sorted(
            (args.geometry_root / split).rglob("*.geometry.npz")
        )
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(
                    _process,
                    source,
                    args.geometry_root,
                    args.output_root,
                    taxonomy,
                    args.overwrite,
                )
                for source in sources
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                row["split"] = split
                rows.append(row)
                if completed % 200 == 0 or completed == len(sources):
                    print(f"[{split}] {completed}/{len(sources)}", flush=True)
    report = {
        "version": "faces-v1",
        "splits": {
            split: sum(row["split"] == split for row in rows)
            for split in args.splits
        },
        "errors": sum(row["status"] == "error" for row in rows),
        "faces": sum(int(row.get("faces") or 0) for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("splits", "errors", "faces")},
            indent=2,
        )
    )
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

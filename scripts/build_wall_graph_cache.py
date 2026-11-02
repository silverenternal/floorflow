#!/usr/bin/env python3
"""Build fpeval-compatible wall graphs from the geometry cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.evaluation.fpeval_bridge import count_double_walls, to_fpeval_plan
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
        default=REPO_ROOT / "outputs/cache/cubicasa5k_wallgraph_v1",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/reports/wall_graph_cache_v1.json",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _process(
    source: Path,
    split: str,
    geometry_root: Path,
    output_root: Path,
    taxonomy: FloorplanTaxonomy,
    overwrite: bool,
) -> dict:
    from fpeval.verify import verify

    relative = source.relative_to(geometry_root / split)
    target = output_root / split / relative.with_suffix(".json")
    if target.is_file() and not overwrite:
        cached = json.loads(target.read_text(encoding="utf-8"))
        return {
            "case": str(relative),
            "split": split,
            "status": "reused",
            "valid": cached.get("valid"),
            "watertight": cached.get("watertight"),
            "walls": None,
            "rooms": None,
            "openings": None,
            "room_source": cached.get("room_source"),
        }
    try:
        raster_plan = wall_plan_from_cache(source, taxonomy)
        plan = to_fpeval_plan(raster_plan)
        result = verify(plan)
        _write_json(
            target,
            {
                **raster_plan.to_json(),
                "source": str(source.relative_to(geometry_root)),
                "cache_version": "wallgraph-v1",
                "valid": result.valid,
                "watertight": result.watertight,
                "verify_issues": result.issues,
            },
        )
        return {
            "case": str(relative),
            "split": split,
            "status": "written",
            "valid": result.valid,
            "watertight": result.watertight,
            "walls": len(plan.walls),
            "rooms": len(plan.rooms),
            "openings": len(plan.openings),
            "room_source": raster_plan.room_source,
            "double_walls": count_double_walls(plan),
            "issues": result.issues,
        }
    except Exception as error:
        return {
            "case": str(relative),
            "split": split,
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
                    split,
                    args.geometry_root,
                    args.output_root,
                    taxonomy,
                    args.overwrite,
                )
                for source in sources
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if completed % 100 == 0 or completed == len(sources):
                    print(f"[{split}] {completed}/{len(sources)}", flush=True)
    written = [row for row in rows if row["status"] == "written"]
    evaluated = [
        row
        for row in rows
        if row["status"] in {"written", "reused"} and row.get("valid") is not None
    ]
    report = {
        "cache_version": "wallgraph-v1",
        "geometry_root": str(args.geometry_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "splits": {
            split: sum(row["split"] == split for row in rows)
            for split in args.splits
        },
        "written": len(written),
        "reused": sum(row["status"] == "reused" for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "valid_fraction": (
            sum(bool(row["valid"]) for row in evaluated) / max(1, len(evaluated))
        ),
        "watertight_fraction": (
            sum(bool(row["watertight"]) for row in evaluated)
            / max(1, len(evaluated))
        ),
        "double_walls_written": sum(
            int(row.get("double_walls") or 0) for row in written
        ),
        "room_source_counts": {
            source: sum(row.get("room_source") == source for row in rows)
            for source in ("graph_faces", "dense_reconciled_faces")
        },
        "rows": sorted(rows, key=lambda row: (row["split"], row["case"])),
    }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "splits",
                    "written",
                    "reused",
                    "errors",
                    "valid_fraction",
                    "watertight_fraction",
                    "double_walls_written",
                )
            },
            indent=2,
        )
    )
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build aligned CubiCasa room-box caches for the chip flow backbone."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.chip_flow.cache import write_packed_split
from floorflow.chip_flow.house_tokens import room_box_case
from floorflow.data.taxonomy import FloorplanTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_v2",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/cache/house_room_tokens_v1",
    )
    parser.add_argument("--max-rooms", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def split_paths(root: Path, split: str) -> list[Path]:
    return sorted((root / split).glob("*/*.npz"))


def build_split(
    paths: list[Path],
    taxonomy: FloorplanTaxonomy,
    max_rooms: int,
) -> list[dict]:
    cases = []
    for path in paths:
        with np.load(path) as payload:
            case = room_box_case(
                payload["target"],
                payload["footprint"],
                taxonomy,
                max_rooms=max_rooms,
                source=str(path),
            )
        if case is not None:
            cases.append(case)
    return cases


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    manifest = {"status": "complete", "splits": {}}
    for split in ("train", "val", "test"):
        paths = split_paths(args.cache_root, split)
        if args.limit > 0:
            paths = paths[: args.limit]
        cases = build_split(paths, taxonomy, args.max_rooms)
        output = args.output / split
        metadata = write_packed_split(cases, output)
        manifest["splits"][split] = {
            "inputs": len(paths),
            "cases": len(cases),
            "block_count_histogram": metadata["block_count_histogram"],
        }
        print(json.dumps({split: manifest["splits"][split]}), flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

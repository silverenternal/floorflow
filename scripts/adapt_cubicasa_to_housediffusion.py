#!/usr/bin/env python3
"""Export CubiCasa room components as an oracle-program HouseDiffusion dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.external_program_adapter import (
    write_housegan_oracle_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/external_baselines/cubicasa_oracle/housediffusion"),
    )
    parser.add_argument("--min-area", type=int, default=8)
    args = parser.parse_args()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    manifest = write_housegan_oracle_dataset(
        dataset,
        args.output_dir,
        min_area=args.min_area,
        list_name="list.txt",
        list_relative=True,
        list_directory=args.output_dir / "json",
    )
    manifest["format"] = "housediffusion_rplan_json"
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

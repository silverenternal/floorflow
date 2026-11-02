#!/usr/bin/env python3
"""Verify that every canonical wall-candidate case has a nonempty cache file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset

from floorflow.training.cad_wall_candidate_epoch_cache import candidate_cache_path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-checkpoint", type=Path, required=True)
    parser.add_argument("--predicted-root", type=Path, required=True)
    parser.add_argument("--candidate-cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/cad_wall_candidate_cache_audit_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.layer_checkpoint, map_location="cpu", weights_only=False)
    image_size = int(payload["config"]["image_size"])
    report = {}
    for split in args.splits:
        dataset = RasterPredictedControlDataset(
            args.predicted_root,
            split,
            args.cache_root,
            args.data_root,
            image_size=image_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        names = []
        for _target, _footprint, _floor_ids, batch_names, _image in loader:
            names.extend(str(name) for name in batch_names)
        missing = []
        empty = []
        for name in names:
            path = candidate_cache_path(
                args.candidate_cache_root,
                split,
                name,
            )
            if not path.is_file():
                missing.append(name)
            elif path.stat().st_size <= 0:
                empty.append(name)
        disk_files = list((args.candidate_cache_root / split).glob("*.pt"))
        report[split] = {
            "expected_cases": len(names),
            "unique_expected_cases": len(set(names)),
            "cache_files": len(disk_files),
            "missing_count": len(missing),
            "empty_count": len(empty),
            "missing": missing,
            "empty": empty,
            "complete": (
                len(names) == len(set(names))
                and not missing
                and not empty
                and len(disk_files) >= len(names)
            ),
        }
    output = {
        "version": "cad-wall-candidate-cache-audit-v1",
        "candidate_cache_root": str(args.candidate_cache_root),
        "all_complete": all(row["complete"] for row in report.values()),
        "splits": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["complete"] for key, value in report.items()}, indent=2))


if __name__ == "__main__":
    main()

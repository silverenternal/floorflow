#!/usr/bin/env python3
"""Repack an object cache into mmap-friendly FloorSet split directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.chip_flow.cache import write_packed_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v1.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "outputs/reports/chip_floorset_packed_v2.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    train = write_packed_split(payload["train"], args.out / "train")
    dev = write_packed_split(payload["dev"], args.out / "dev")
    manifest = {
        "status": "complete",
        "source": str(args.source),
        "out": str(args.out),
        "train": train,
        "dev": dev,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": manifest["status"],
        "out": manifest["out"],
        "train_cases": train["cases"],
        "dev_cases": dev["cases"],
        "train_blocks": [train["min_blocks"], train["max_blocks"]],
        "dev_blocks": [dev["min_blocks"], dev["max_blocks"]],
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

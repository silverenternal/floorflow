#!/usr/bin/env python3
"""Build deterministic train/dev caches from the raw 10-column FloorSet shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.chip_flow.dataset import (
    index_legacy_shards,
    list_legacy_shards,
    materialize_cases,
    select_legacy_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=REPO_ROOT / "third_party/Trinity/data/floorset_lite",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v1.pt",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "outputs/reports/chip_floorset_split_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20090220)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--train-min-blocks", type=int, default=21)
    parser.add_argument("--train-max-blocks", type=int, default=60)
    parser.add_argument("--dev-per-count", type=int, default=100)
    parser.add_argument("--dev-random", type=int, default=2000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    paths = list_legacy_shards(args.raw_root)
    block_counts, layout_counts = index_legacy_shards(paths)
    train_entries, dev_entries, split = select_legacy_splits(
        block_counts,
        seed=args.seed,
        dev_per_count=args.dev_per_count,
        dev_random=args.dev_random,
        train_size=args.train_size,
        train_min_blocks=args.train_min_blocks,
        train_max_blocks=args.train_max_blocks,
    )
    train, dev = materialize_cases(paths, train_entries, dev_entries)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": train, "dev": dev}, args.cache)

    manifest = {
        "status": "complete",
        "raw_root": str(args.raw_root),
        "cache": str(args.cache),
        "cache_sha256": sha256(args.cache),
        "shards": len(paths),
        "layouts_indexed": sum(layout_counts),
        "block_count_histogram": {
            str(count): total
            for count, total in sorted(
                sum(
                    (
                        Counter({block_count: layout_count})
                        for block_count, layout_count in zip(
                            block_counts,
                            layout_counts,
                            strict=True,
                        )
                    ),
                    start=Counter(),
                ).items()
            )
        },
        **split,
        "train_cases": len(train),
        "dev_cases": len(dev),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

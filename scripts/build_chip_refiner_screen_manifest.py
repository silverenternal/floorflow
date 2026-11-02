#!/usr/bin/env python3
"""Freeze a deterministic block-count-stratified refiner development subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from floorflow.chip_flow.cache import PackedCaseStore

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev",
    )
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--salt", default="chip-refiner-sota-v44")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_refiner_screen_manifest_v44.json",
    )
    return parser.parse_args()


def stable_rank(count: int, salt: str) -> int:
    payload = f"{salt}:{count}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def main() -> None:
    args = parse_args()
    cache = args.cache.resolve()
    output = args.output.resolve()
    store = PackedCaseStore(cache)
    counts = sorted(store.by_block_count)
    if not 1 <= args.cases <= len(counts):
        raise ValueError("--cases must be between 1 and the number of block counts")
    positions = np.linspace(0, len(counts) - 1, args.cases)
    selected_counts = [counts[round(position)] for position in positions]
    if len(set(selected_counts)) != args.cases:
        raise ValueError("stratified selection did not produce unique block counts")
    rows = []
    for count in selected_counts:
        indices = sorted(int(index) for index in store.indices(count))
        selected = indices[stable_rank(count, args.salt) % len(indices)]
        rows.append(
            {
                "index": selected,
                "block_count": count,
                "candidates_in_stratum": len(indices),
            }
        )
    metadata = cache / "metadata.json"
    report = {
        "version": "chip-refiner-screen-manifest-v44",
        "cache": str(cache.relative_to(ROOT)),
        "cache_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "salt": args.salt,
        "selection_rule": (
            "Uniformly span sorted block counts 21..120, then choose one case per "
            "stratum by SHA256(salt:block_count) modulo the sorted index list. No "
            "model output or target metric is read."
        ),
        "cases": len(rows),
        "indices": [row["index"] for row in rows],
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

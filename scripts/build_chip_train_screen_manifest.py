#!/usr/bin/env python3
"""Build a deterministic, block-count-stratified training subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from floorflow.chip_flow.cache import PackedCaseStore

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs/chip_train_screen_5k_v1.json",
    )
    return parser.parse_args()


def allocate_quotas(counts: dict[int, int], limit: int) -> dict[int, int]:
    total = sum(counts.values())
    exact = {count: limit * size / total for count, size in counts.items()}
    quotas = {
        count: min(counts[count], int(np.floor(value)))
        for count, value in exact.items()
    }
    remaining = limit - sum(quotas.values())
    order = sorted(exact, key=lambda count: (exact[count] - quotas[count], -count), reverse=True)
    while remaining > 0:
        progressed = False
        for count in order:
            if quotas[count] < counts[count]:
                quotas[count] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("subset limit exceeds available cases")
    while remaining < 0:
        progressed = False
        for count in sorted(order, reverse=True):
            if quotas[count] > 0:
                quotas[count] -= 1
                remaining += 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("cannot allocate subset quotas")
    return quotas


def main() -> None:
    args = parse_args()
    cache = args.cache.resolve()
    output = args.output.resolve()
    store = PackedCaseStore(cache)
    counts = {
        int(count): len(indices)
        for count, indices in store.by_block_count.items()
    }
    quotas = allocate_quotas(counts, args.limit)
    rng = np.random.default_rng(args.seed)
    selected = []
    for count in sorted(store.by_block_count):
        shuffled = rng.permutation(store.indices(count))
        selected.extend(int(index) for index in shuffled[: quotas[count]])
    selected = sorted(selected)
    selected_counts = Counter(int(store.arrays["n"][index]) for index in selected)
    source_counts = Counter(int(store.arrays["source_id"][index]) for index in selected)
    all_sources = Counter(int(value) for value in store.arrays["source_id"])
    metadata_path = cache / "metadata.json"
    report = {
        "version": args.version or f"chip-train-screen-{args.limit}-v1",
        "purpose": "Deterministic block-count-stratified training subset.",
        "cache": str(cache.relative_to(ROOT)),
        "cache_metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
        "seed": args.seed,
        "indices": selected,
        "case_count": len(selected),
        "block_count_histogram": {
            str(count): selected_counts[count] for count in sorted(selected_counts)
        },
        "source_family_histogram": {
            store.metadata["source_paths"][source]: {
                "selected": source_counts[source],
                "available": all_sources[source],
            }
            for source in sorted(all_sources)
        },
        "policy": (
            "Frozen deterministic stratified subset for mechanism screening. "
            "A validated mechanism must be registered before full-data training."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "case_count": len(selected),
                "block_counts": dict(sorted(selected_counts.items())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

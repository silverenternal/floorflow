#!/usr/bin/env python3
"""Validate the 21-120 chip training cache and unchanged dev split."""

from __future__ import annotations

import argparse
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
        default=ROOT / "outputs/cache/chip_floorset_200k_train_12k_dev_range_v1",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_floorset_range_cache_audit_v1.json",
    )
    parser.add_argument("--train-indices", type=Path, default=None)
    return parser.parse_args()


def keys(store: PackedCaseStore) -> set[tuple[str, int]]:
    paths = np.asarray(store.metadata["source_paths"])
    source_ids = np.asarray(store.arrays["source_id"])
    source_indices = np.asarray(store.arrays["source_index"])
    return {
        (
            str(
                (
                    Path(paths[source_id])
                    if Path(paths[source_id]).is_absolute()
                    else ROOT / Path(paths[source_id])
                ).resolve()
            ),
            int(source_index),
        )
        for source_id, source_index in zip(source_ids, source_indices)
    }


def main() -> None:
    args = parse_args()
    train = PackedCaseStore(args.cache / "train")
    dev = PackedCaseStore(args.cache / "dev")
    reference_dev = PackedCaseStore(args.reference / "dev")
    train_keys = keys(train)
    dev_keys = keys(dev)
    reference_keys = keys(reference_dev)
    counts = sorted(int(count) for count in train.by_block_count)
    dev_same = {}
    for name in ("areas", "boxes", "constraints", "features"):
        dev_same[name] = bool(
            np.array_equal(
                np.asarray(dev.arrays[name]),
                np.asarray(reference_dev.arrays[name]),
            )
        )
    checks = {
        "train_cases_200k": len(train) == 200000,
        "dev_cases_12k": len(dev) == 12000,
        "train_range_21_120": train.metadata["min_blocks"] == 21
        and train.metadata["max_blocks"] == 120,
        "all_block_counts_present": counts == list(range(21, 121)),
        "train_dev_disjoint": not bool(train_keys & dev_keys),
        "dev_matches_reference": all(dev_same.values())
        and dev_keys == reference_keys,
    }
    selected = None
    if args.train_indices is not None:
        payload = json.loads(args.train_indices.read_text(encoding="utf-8"))
        values = payload.get("indices", payload) if isinstance(payload, dict) else payload
        selected_indices = {int(value) for value in values}
        counts_by_block = Counter(
            {
                int(count): sum(
                    1
                    for index in train.indices(count)
                    if int(index) in selected_indices
                )
                for count in train.by_block_count
            }
        )
        selected_cases = sum(counts_by_block.values())
        present_counts = [
            count for count, value in counts_by_block.items() if value
        ]
        if not present_counts:
            raise ValueError("selected train subset is empty")
        selected = {
            "path": str(args.train_indices),
            "cases": selected_cases,
            "block_count_groups": sum(value > 0 for value in counts_by_block.values()),
            "blocks": [
                min(present_counts),
                max(present_counts),
            ],
            "buckets": {
                "21_60": sum(
                    value
                    for count, value in counts_by_block.items()
                    if 21 <= count <= 60
                ),
                "61_90": sum(
                    value
                    for count, value in counts_by_block.items()
                    if 61 <= count <= 90
                ),
                "91_120": sum(
                    value
                    for count, value in counts_by_block.items()
                    if 91 <= count <= 120
                ),
            },
        }
        checks["selected_subset_cases_expected"] = selected_cases == len(
            selected_indices
        )
        checks["selected_subset_all_blocks_present"] = all(
            value > 0 for value in counts_by_block.values()
        )
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "cache": str(args.cache),
        "train": {
            "cases": len(train),
            "blocks": [
                train.metadata["min_blocks"],
                train.metadata["max_blocks"],
            ],
            "unique_block_counts": len(counts),
        },
        "dev": {
            "cases": len(dev),
            "blocks": [
                dev.metadata["min_blocks"],
                dev.metadata["max_blocks"],
            ],
            "tensor_equality": dev_same,
        },
        "train_dev_overlap": len(train_keys & dev_keys),
        "selected_subset": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

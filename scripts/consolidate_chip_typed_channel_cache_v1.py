#!/usr/bin/env python3
"""Consolidate per-case typed channel states into block-count shards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.training_manifest import load_train_subset
from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.chip_flow.typed_channel_cache import (
    read_typed_channel_state,
    typed_channel_case_path,
    write_typed_channel_shard,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--train-indices", type=Path, required=True)
    args = parser.parse_args()
    store = PackedCaseStore(args.data_root / args.split)
    indices = sorted(load_train_subset(args.train_indices))
    by_count: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        by_count[int(store.arrays["n"][index])].append(index)
    source_manifest_path = args.source / args.split / "manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.is_file()
        else {}
    )
    shards = []
    for count, group in sorted(by_count.items()):
        states = [
            read_typed_channel_state(
                typed_channel_case_path(args.source, args.split, index),
                device="cpu",
            )[0]
            for index in group
        ]
        shard_path = write_typed_channel_shard(
            args.output,
            args.split,
            count,
            group,
            states,
            metadata={
                "source": str(args.source),
                "encoder_fingerprint": source_manifest.get(
                    "encoder_fingerprint"
                ),
            },
        )
        shards.append(
            {
                "block_count": count,
                "cases": len(group),
                "case_ids": [str(index) for index in group],
                "path": str(shard_path),
            }
        )
    write_json_atomic(
        args.output / args.split / "shards.json",
        {
            "version": "chip-typed-channel-shards-v1",
            "source": str(args.source),
            "split": args.split,
            "case_count": len(indices),
            "encoder_fingerprint": source_manifest.get(
                "encoder_fingerprint"
            ),
            "shards": shards,
        },
    )
    print(
        json.dumps(
            {
                "cases": len(indices),
                "shards": len(shards),
                "output": str(args.output / args.split),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

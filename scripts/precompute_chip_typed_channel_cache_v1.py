#!/usr/bin/env python3
"""Precompute frozen pre-fusion typed channel states for packed chip cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchy_context import encode_typed_channel_state
from floorflow.chip_flow.training_manifest import load_train_subset
from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.chip_flow.typed_channel_cache import (
    cache_key,
    encoder_state_fingerprint,
    input_fingerprint,
    typed_channel_case_path,
    write_typed_channel_state,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = load_chip_checkpoint(checkpoint, device, use_ema=True).eval()
    if not hasattr(model, "typed"):
        raise TypeError("typed channel cache requires a typed hierarchy model")
    store = PackedCaseStore(args.data_root / args.split)
    indices = store.indices()
    if args.indices is not None:
        selected = load_train_subset(args.indices)
        indices = np.asarray(sorted(selected), dtype=np.int64)
    elif args.limit:
        indices = indices[: args.limit]
    if indices.size and (indices.min() < 0 or indices.max() >= len(store)):
        raise ValueError("training subset contains out-of-range case indices")
    encoder_fingerprint = encoder_state_fingerprint(model.typed)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    manifest = {
        "version": "chip-typed-channel-cache-v1",
        "data_root": str(args.data_root),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "encoder_fingerprint": encoder_fingerprint,
        "case_count": len(indices),
        "cases": [],
    }
    with torch.no_grad():
        for position, index in enumerate(indices, start=1):
            batch = store.batch(np.asarray([index]), device=device)
            state = encode_typed_channel_state(model, batch)
            sample_fingerprint = input_fingerprint(
                {
                    key: value
                    for key, value in batch.items()
                    if torch.is_tensor(value)
                }
            )
            path = typed_channel_case_path(args.output, args.split, int(index))
            write_typed_channel_state(
                path,
                state,
                metadata={
                    "case_index": int(index),
                    "sample_fingerprint": sample_fingerprint,
                    "key": cache_key(encoder_fingerprint, sample_fingerprint),
                },
            )
            manifest["cases"].append(
                {
                    "index": int(index),
                    "path": str(path),
                    "key": cache_key(encoder_fingerprint, sample_fingerprint),
                }
            )
            if position % 500 == 0:
                print(
                    json.dumps(
                        {"cached": position, "total": len(indices)}
                    ),
                    flush=True,
                )
    write_json_atomic(args.output / args.split / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "cases": len(indices),
                "output": str(args.output / args.split),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

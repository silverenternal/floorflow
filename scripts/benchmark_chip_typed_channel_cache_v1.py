#!/usr/bin/env python3
"""Benchmark online channel encoding against the frozen-state cache."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.training_manifest import load_train_subset
from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.chip_flow.training_scopes import (
    apply_training_scope,
    attach_typed_channel_cache,
    parse_training_scope,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--train-indices", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/chip_channel_cache_training_step_speed_v1.json"),
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batches <= 0:
        raise ValueError("batch size and batch count must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = load_chip_checkpoint(checkpoint, device, use_ema=True).eval()
    apply_training_scope(model, parse_training_scope("fusion"))
    store = PackedCaseStore(args.data_root / args.split)
    selected = sorted(load_train_subset(args.train_indices))
    counts = store.arrays["n"][np.asarray(selected)]
    count = int(statistics.mode(counts.tolist()))
    pool = [
        index for index in selected if int(store.arrays["n"][index]) == count
    ]
    required = args.batch_size * args.batches
    if len(pool) < required:
        raise ValueError("not enough same-count cases for the requested benchmark")
    batches = [
        np.asarray(pool[offset : offset + args.batch_size], dtype=np.int64)
        for offset in range(0, required, args.batch_size)
    ]
    online = []
    cached = []
    maximum_error = 0.0
    for indices in batches:
        batch = store.batch(indices, device=device)
        batch["_indices"] = indices
        model.typed_channel_cache_root = None
        model.zero_grad(set_to_none=True)
        _synchronize(device)
        started = time.perf_counter()
        online_context = model.prepare_context(batch)
        online_loss = online_context.typed.block_tokens.float().square().mean()
        online_loss.backward()
        _synchronize(device)
        online.append(time.perf_counter() - started)
        attach_typed_channel_cache(
            model,
            SimpleNamespace(
                typed_channel_cache_root=str(args.cache_root),
                typed_channel_cache_split=args.split,
            ),
        )
        model.zero_grad(set_to_none=True)
        _synchronize(device)
        started = time.perf_counter()
        cached_context = model.prepare_context(batch)
        cached_loss = cached_context.typed.block_tokens.float().square().mean()
        cached_loss.backward()
        _synchronize(device)
        cached.append(time.perf_counter() - started)
        maximum_error = max(
            maximum_error,
            float(
                (
                    online_context.typed.block_tokens
                    - cached_context.typed.block_tokens
                ).abs().max().detach()
            ),
        )
    online_median = statistics.median(online)
    cached_median = statistics.median(cached)
    report = {
        "version": "chip-channel-cache-training-step-speed-v1",
        "device": str(device),
        "batch_size": args.batch_size,
        "block_count": count,
        "batches": len(batches),
        "online_median_seconds": online_median,
        "cached_median_seconds": cached_median,
        "speedup": online_median / max(cached_median, 1.0e-12),
        "maximum_output_error": maximum_error,
        "scope": "fusion",
        "evidence_boundary": (
            "Typed-context step only; full backbone/data-loader end-to-end "
            "speed remains separately measured."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

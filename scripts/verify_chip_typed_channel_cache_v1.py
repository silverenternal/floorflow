#!/usr/bin/env python3
"""Verify a real typed-channel cache against online encoding."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchy_context import encode_typed_channel_state
from floorflow.chip_flow.training_manifest import load_train_subset
from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.chip_flow.typed_channel_cache import (
    load_typed_channel_states,
    read_typed_channel_state,
    typed_channel_case_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--indices", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=4.0e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/chip_channel_cache_real_equivalence_v1.json"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = load_chip_checkpoint(checkpoint, device, use_ema=True).eval()
    store = PackedCaseStore(args.data_root / args.split)
    indices = store.indices()
    if args.indices is not None:
        indices = np.asarray(sorted(load_train_subset(args.indices)), dtype=np.int64)
    elif args.limit:
        indices = indices[: args.limit]
    by_count: dict[int, list[int]] = defaultdict(list)
    maximum_channel_error = 0.0
    maximum_fused_error = 0.0
    if args.tolerance <= 0.0:
        raise ValueError("--tolerance must be positive")
    tolerance = float(args.tolerance)
    groups = []
    individual_fused: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for index in indices:
            batch = store.batch(np.asarray([index]), device=device)
            online = encode_typed_channel_state(model, batch)
            cached, _metadata = read_typed_channel_state(
                typed_channel_case_path(args.cache_root, args.split, int(index)),
                device=device,
            )
            channel_error = max(
                float(
                    (
                        getattr(online, name)
                        - getattr(cached, name)
                    ).abs().max()
                )
                for name in (
                    "block_tokens",
                    "global_token",
                    "structural_tokens",
                    "constraint_tokens",
                    "port_tokens",
                )
            )
            online_fused = model.typed.fuse_channels(online).block_tokens
            cached_fused = model.typed.fuse_channels(cached).block_tokens
            fused_error = float((online_fused - cached_fused).abs().max())
            maximum_channel_error = max(maximum_channel_error, channel_error)
            maximum_fused_error = max(maximum_fused_error, fused_error)
            by_count[int(store.arrays["n"][index])].append(int(index))
            individual_fused[int(index)] = online_fused
        for count, group_indices in by_count.items():
            collated = load_typed_channel_states(
                args.cache_root,
                args.split,
                group_indices,
                device=device,
            )
            collated_fused = model.typed.fuse_channels(collated).block_tokens
            individual = [individual_fused[index] for index in group_indices]
            group_error = float(
                (collated_fused - torch.cat(individual)).abs().max()
            )
            groups.append(
                {
                    "block_count": count,
                    "cases": len(group_indices),
                    "collation_max_abs_error": group_error,
                }
            )
    report = {
        "version": "chip-channel-cache-real-equivalence-v1",
        "cases": len(indices),
        "split": args.split,
        "maximum_channel_error": maximum_channel_error,
        "maximum_fused_error": maximum_fused_error,
        "group_collation": groups,
        "tolerance": tolerance,
        "strict_bitwise_equivalent": (
            maximum_channel_error == 0.0 and maximum_fused_error == 0.0
        ),
        "equivalent": (
            maximum_channel_error <= tolerance
            and maximum_fused_error <= tolerance
            and all(
                group["collation_max_abs_error"] <= tolerance
                for group in groups
            )
        ),
        "evidence_boundary": (
            "Frozen-subset cases verify cache exactness and collation; a real "
            "training-step speedup remains separate evidence."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

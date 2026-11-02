#!/usr/bin/env python3
"""Measure exact typed-channel cache equivalence and frozen-fusion latency."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.chip_flow.typed_channel_cache import (
    cache_key,
    encoder_state_fingerprint,
    input_fingerprint,
    read_typed_channel_state,
    write_typed_channel_state,
)
from floorflow.chip_flow.typed_graph_encoder import TypedGraphEncoder


def _inputs() -> tuple[torch.Tensor, torch.Tensor, dict]:
    batch, count, pins = 4, 16, 24
    constraints = torch.zeros(batch, count, 5)
    constraints[:, :4, 2] = 1.0
    constraints[:, :, 3] = torch.arange(count) % 4 + 1
    constraints[:, :, 4] = (torch.arange(count) % 15 + 1).float()
    pairs = count * 3
    source = torch.arange(pairs) % count
    target = (source + 1 + torch.arange(pairs) // count) % count
    graph = {
        "areas": torch.rand(batch, count) + 0.5,
        "b2b": torch.stack(
            [source.float(), target.float(), torch.rand(pairs) + 0.1],
            dim=-1,
        )[None].expand(batch, -1, -1),
        "p2b": torch.stack(
            [
                torch.arange(pins).float(),
                (torch.arange(pins) % count).float(),
                torch.rand(pins) + 0.1,
            ],
            dim=-1,
        )[None].expand(batch, -1, -1),
        "pins": torch.rand(batch, pins, 2),
        "n_b2b": torch.full((batch,), pairs),
        "n_p2b": torch.full((batch,), pins),
        "n_pins": torch.full((batch,), pins),
        "global_spec": torch.randn(batch, 7),
        "node_features": torch.randn(batch, count, 4),
        "port_features": torch.randn(batch, pins, 4),
    }
    return torch.randn(batch, count, 19), constraints, graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--cache",
        default="outputs/cache/chip_channel_cache_audit_v1/state.npz",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/chip_channel_cache_equivalence_v1.json",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    torch.manual_seed(31)
    encoder = TypedGraphEncoder(
        hidden=64,
        heads=4,
        graph_layers=2,
        relation_specific=True,
        global_spec_conditioning=True,
        node_semantic_dim=4,
        port_semantic_dim=4,
    ).eval()
    features, constraints, graph = _inputs()
    state = encoder.encode_channels(features, constraints, **graph)
    encoder_fp = encoder_state_fingerprint(encoder)
    inputs_fp = input_fingerprint(
        {
            "features": features,
            "constraints": constraints,
            **graph,
        }
    )
    write_typed_channel_state(
        Path(args.cache),
        state,
        metadata={
            "case": "cache-audit",
            "encoder_fingerprint": encoder_fp,
            "input_fingerprint": inputs_fp,
            "key": cache_key(encoder_fp, inputs_fp),
        },
    )
    cache_path = Path(args.cache)
    restored, metadata = read_typed_channel_state(cache_path, device="cpu")
    online = encoder.fuse_channels(state).block_tokens
    cached = encoder.fuse_channels(restored).block_tokens
    max_error = float((online - cached).abs().max().detach())
    online_times = []
    cached_times = []
    with torch.no_grad():
        for _ in range(args.repeats + 1):
            started = time.perf_counter()
            encoded = encoder.encode_channels(features, constraints, **graph)
            encoder.fuse_channels(encoded)
            online_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            cached_state, _ = read_typed_channel_state(
                cache_path,
                device="cpu",
            )
            encoder.fuse_channels(cached_state)
            cached_times.append(time.perf_counter() - started)
    online_median = statistics.median(online_times[1:])
    cached_median = statistics.median(cached_times[1:])
    report = {
        "version": "chip-channel-cache-equivalence-v1",
        "cache": args.cache,
        "metadata": metadata,
        "max_abs_error": max_error,
        "equivalent": max_error == 0.0,
        "online_median_seconds": online_median,
        "cached_median_seconds": cached_median,
        "speedup": online_median / max(cached_median, 1.0e-12),
        "evidence_boundary": (
            "Synthetic variable-batch codec probe only; no real training cache "
            "has been integrated."
        ),
    }
    write_json_atomic(args.output, report)
    print(report)


if __name__ == "__main__":
    main()

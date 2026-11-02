#!/usr/bin/env python3
"""Benchmark exact CPU matching against GPU Sinkhorn for room-query sizes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from floorflow.objectives.gpu_assignment import (
    exact_assignment_batch,
    sinkhorn_log_assignment,
)


def _time_exact(
    batch: int,
    queries: int,
    rooms: int,
    repeats: int,
) -> float:
    costs = [
        torch.rand(queries, rooms)
        for _ in range(batch)
    ]
    for _ in range(5):
        exact_assignment_batch(costs)
    started = time.perf_counter()
    for _ in range(repeats):
        exact_assignment_batch(costs)
    return (time.perf_counter() - started) / repeats * 1.0e3


def _time_sinkhorn(
    batch: int,
    queries: int,
    rooms: int,
    repeats: int,
    device: torch.device,
) -> float:
    costs = torch.rand(batch, queries, rooms, device=device)
    for _ in range(5):
        sinkhorn_log_assignment(costs)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        sinkhorn_log_assignment(costs)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) / repeats * 1.0e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--rooms", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/reports/typed_channel_assignment_microbench_v1.json"
        ),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested without an available GPU")
    exact_ms = _time_exact(
        args.batch,
        args.queries,
        args.rooms,
        args.repeats,
    )
    sinkhorn_ms = _time_sinkhorn(
        args.batch,
        args.queries,
        args.rooms,
        args.repeats,
        device,
    )
    report = {
        "schema_version": "typed-channel-assignment-microbench-v1",
        "device": str(device),
        "shape": [args.batch, args.queries, args.rooms],
        "repeats": args.repeats,
        "exact_cpu_ms": exact_ms,
        "gpu_sinkhorn_ms": sinkhorn_ms,
        "gpu_to_cpu_ratio": sinkhorn_ms / exact_ms,
        "decision": (
            "retain_exact_cpu_matching"
            if sinkhorn_ms >= exact_ms
            else "consider_gpu_matching"
        ),
        "scope": (
            "Small query matrices only. This benchmark isolates matching "
            "cost and does not include model or data-loader latency."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

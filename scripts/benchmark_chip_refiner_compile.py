#!/usr/bin/env python3
"""Measure compiled/eager closed-form refiner equivalence and throughput."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import PAPER_WEIGHTS, refine_closed

from floorflow.chip_flow.cache import PackedCaseStore
from scripts.evaluate_chip_refiner import make_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_refiner_compile_benchmark_v1.json",
    )
    return parser.parse_args()


@torch.no_grad()
def run(
    latent: torch.Tensor,
    case,
    steps: int,
    chunk: int,
    *,
    compile_loop: bool,
) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize(latent.device)
    started = time.perf_counter()
    refined = refine_closed(
        latent,
        case,
        PAPER_WEIGHTS,
        steps,
        1.0e-3,
        "linear",
        snapshots=(steps,),
        chunk=chunk,
        compile_loop=compile_loop,
    )[steps]
    torch.cuda.synchronize(latent.device)
    return refined, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    store = PackedCaseStore(args.cache)
    block_count = min(store.by_block_count)
    batch = store.batch(store.indices(block_count)[:4], device=device)
    case = build_refine_case(make_instances(batch), 1, device)
    latent = batch["target_latent"].float()
    run(latent, case, args.chunk, args.chunk, compile_loop=False)
    eager, eager_seconds = run(
        latent,
        case,
        args.steps,
        args.chunk,
        compile_loop=False,
    )
    run(latent, case, args.chunk, args.chunk, compile_loop=True)
    compiled_first, compiled_first_seconds = run(
        latent,
        case,
        args.steps,
        args.chunk,
        compile_loop=True,
    )
    compiled_warm, compiled_warm_seconds = run(
        latent,
        case,
        args.steps,
        args.chunk,
        compile_loop=True,
    )
    max_difference = float((eager - compiled_first).abs().max())
    checks = {
        "finite": bool(torch.isfinite(compiled_first).all()),
        "numerically_equivalent": max_difference <= 1.0e-4,
        "warm_output_stable": float((compiled_first - compiled_warm).abs().max())
        == 0.0,
    }
    report = {
        "version": "chip-refiner-compile-benchmark-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "device": str(device),
        "block_count": block_count,
        "cases": int(latent.shape[0]),
        "steps": args.steps,
        "chunk": args.chunk,
        "eager_seconds": eager_seconds,
        "compiled_first_seconds": compiled_first_seconds,
        "compiled_warm_seconds": compiled_warm_seconds,
        "compiled_warm_speedup": eager_seconds / compiled_warm_seconds,
        "max_abs_difference": max_difference,
        "checks": checks,
        "scope": "Refiner-only plumbing benchmark; no placement-quality claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

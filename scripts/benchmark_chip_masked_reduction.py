#!/usr/bin/env python3
"""Benchmark the reusable masked token reduction used by typed hierarchy paths."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.sparse_layers import masked_token_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_masked_reduction_benchmark.json",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(
    fn,
    device: torch.device,
    warmup: int,
    trials: int,
) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(trials):
        started = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - started) * 1.0e3)
    return float(np.median(samples))


def reference_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(tokens.dtype)
    total = torch.einsum("bn,bnd->bd", weights, tokens)
    return total / weights.sum(dim=1, keepdim=True).clamp_min(1.0)


def benchmark_shape(
    name: str,
    valid_tokens: int,
    *,
    batch: int,
    hidden: int,
    device: torch.device,
    warmup: int,
    trials: int,
) -> dict:
    tokens = torch.randn(batch, valid_tokens, hidden, device=device)
    mask = torch.rand(batch, valid_tokens, device=device) > 0.15
    reference = reference_mean(tokens, mask)
    vectorized = masked_token_mean(tokens, mask)
    reference_ms = timed(
        lambda: reference_mean(tokens, mask),
        device,
        warmup,
        trials,
    )
    vectorized_ms = timed(
        lambda: masked_token_mean(tokens, mask),
        device,
        warmup,
        trials,
    )
    left = tokens.detach().clone().requires_grad_(True)
    right = tokens.detach().clone().requires_grad_(True)
    reference_mean(left, mask).square().mean().backward()
    masked_token_mean(right, mask).square().mean().backward()
    return {
        "tokens": valid_tokens,
        "einsum_ms": reference_ms,
        "bmm_ms": vectorized_ms,
        "speedup": reference_ms / vectorized_ms,
        "forward_max_abs_difference": float((reference - vectorized).abs().max()),
        "gradient_max_abs_difference": float((left.grad - right.grad).abs().max()),
        "name": name,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    reports = [
        benchmark_shape(
            "block",
            40,
            batch=args.batch_size,
            hidden=args.hidden,
            device=device,
            warmup=args.warmup,
            trials=args.trials,
        ),
        benchmark_shape(
            "structural",
            9,
            batch=args.batch_size,
            hidden=args.hidden,
            device=device,
            warmup=args.warmup,
            trials=args.trials,
        ),
        benchmark_shape(
            "port",
            8,
            batch=args.batch_size,
            hidden=args.hidden,
            device=device,
            warmup=args.warmup,
            trials=args.trials,
        ),
    ]
    report = {
        "version": "chip-v46-masked-reduction-benchmark-v1",
        "status": "pass"
        if all(
            row["forward_max_abs_difference"] <= 1.0e-6
            and row["gradient_max_abs_difference"] <= 1.0e-6
            for row in reports
        )
        else "fail",
        "device": str(device),
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "results": reports,
        "implementation": "torch.bmm(weight[:, None, :], tokens).squeeze(1)",
        "scope": "Operator microbenchmark only; no end-to-end quality claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

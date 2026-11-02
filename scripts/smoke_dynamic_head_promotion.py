#!/usr/bin/env python3
"""CPU contract benchmark for fixed and dynamic head variants."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from floorflow.heads.pixel_distribution import PixelDistributionHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--classes", type=int, default=17)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/dynamic_head_cpu_smoke_v1.json"),
    )
    return parser.parse_args()


@torch.inference_mode()
def benchmark_variant(
    variant: str,
    hidden_channels: int,
    classes: int,
    size: int,
    repeats: int,
    device: torch.device,
) -> dict[str, object]:
    head = PixelDistributionHead(
        hidden_channels,
        classes,
        hidden_channels=hidden_channels,
        room_ids=tuple(range(1, min(classes, 7))),
        dynamic_head_variant=variant,
    ).to(device).eval()
    router = getattr(head.dynamic_output, "router", None)
    if router is not None:
        with torch.no_grad():
            router[-1].weight.normal_(0.0, 0.1)
    features = torch.randn(2, hidden_channels, size, size, device=device)
    for _ in range(2):
        head(features)
    started = time.perf_counter()
    outputs = []
    for _ in range(repeats):
        outputs.append(head(features))
    elapsed = (time.perf_counter() - started) / repeats
    weights = outputs[-1]["multi_scale_weights"]
    route_variation = None
    if weights is not None:
        route_variation = float(weights.std(dim=1).mean())
    diagnostics = outputs[-1]["dynamic_query_diagnostics"]
    return {
        "variant": variant,
        "parameters": sum(parameter.numel() for parameter in head.parameters()),
        "seconds_per_batch": elapsed,
        "logits_shape": list(outputs[-1]["logits"].shape),
        "route_std": route_variation,
        "query_budget_mean": (
            float(diagnostics["query_budget"].mean())
            if diagnostics is not None
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    results = [
        benchmark_variant(
            variant,
            args.hidden_channels,
            args.classes,
            args.size,
            args.repeats,
            device,
        )
        for variant in ("fixed", "mixture", "generated", "query")
    ]
    report = {
        "device": str(device),
        "protocol": {
            "batch": 2,
            "size": args.size,
            "repeats": args.repeats,
            "gpu_training": False,
            "router_perturbed_for_variation_probe": True,
        },
        "variants": results,
        "status": (
            "pass"
            if all(
                not isinstance(value, float) or math.isfinite(value)
                for row in results
                for value in row.values()
            )
            else "fail"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

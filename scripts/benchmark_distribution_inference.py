#!/usr/bin/env python3
"""Measure distribution-head and CFM inference latency and memory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.flow.core import sample_floorplans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _taxonomy, _payload = restore_distribution_model(
        args.checkpoint,
        device,
    )
    model.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    target, footprint, floor_ids, _ = next(
        iter(DataLoader(dataset, batch_size=args.limit, shuffle=False))
    )
    del target
    footprint = footprint[:, None].to(device)
    floor_ids = floor_ids.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def distribution_pass() -> float:
        started = time.perf_counter()
        for _ in range(args.repeats):
            synchronize(device)
            model.contour_initial_state(footprint, floor_ids=floor_ids)
            synchronize(device)
        return time.perf_counter() - started

    def flow_pass(steps: int) -> float:
        started = time.perf_counter()
        for _ in range(args.repeats):
            synchronize(device)
            sample_floorplans(
                model.base,
                footprint[:, 0],
                steps=steps,
                heun=False,
                floor_ids=floor_ids,
            )
            synchronize(device)
        return time.perf_counter() - started

    distribution_pass()
    flow_pass(4)
    distribution_seconds = distribution_pass()
    flow_seconds = {steps: flow_pass(steps) for steps in (4, 16)}
    sample_count = args.limit * args.repeats
    report = {
        "checkpoint": str(args.checkpoint),
        "cases": args.limit,
        "repeats": args.repeats,
        "batch_size": args.limit,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "distribution_head": {
            "ode_steps": 0,
            "samples_per_second": sample_count / distribution_seconds,
            "milliseconds_per_sample": (
                1000.0 * distribution_seconds / sample_count
            ),
        },
        "cfm": {
            str(steps): {
                "ode_steps": steps,
                "samples_per_second": sample_count / seconds,
                "milliseconds_per_sample": 1000.0 * seconds / sample_count,
            }
            for steps, seconds in flow_seconds.items()
        },
        "cuda_peak_allocated_gb": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else None
        ),
        "cuda_peak_reserved_gb": (
            torch.cuda.max_memory_reserved(device) / 1024**3
            if device.type == "cuda"
            else None
        ),
        "best_of_n_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit finite proxy guidance on a real chip-flow checkpoint and batch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.guided_sampling import sample_hierarchy_guided_flow

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev",
    )
    parser.add_argument("--block-count", type=int, default=31)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.02)
    parser.add_argument("--guidance-energy", choices=("routing", "composite"), default="routing")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_a15_energy_guidance_smoke.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    store = PackedCaseStore(args.cache)
    indices = store.indices(args.block_count)[: args.cases]
    batch = store.batch(indices, device="cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    started = time.perf_counter()
    latent, diagnostics = sample_hierarchy_guided_flow(
        model,
        batch,
        sampling_steps=args.sampling_steps,
        generator=torch.Generator().manual_seed(19),
        guidance_scale=args.guidance_scale,
        energy_mode=args.guidance_energy,
    )
    elapsed = time.perf_counter() - started
    checks = {
        "finite_latent": bool(torch.isfinite(latent).all()),
        "positive_gradient_rms": float(diagnostics["guidance_gradient_rms_mean"])
        > 0.0,
        "positive_update_rms": float(diagnostics["guidance_update_rms_mean"]) > 0.0,
        "finite_energy": bool(torch.isfinite(diagnostics["guidance_energy_mean"])),
    }
    report = {
        "version": "chip-a15-guided-sampling-smoke-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "CPU real-checkpoint proxy-gradient smoke; no quality claim",
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "cases": int(latent.shape[0]),
        "blocks": int(latent.shape[1]),
        "sampling_steps": args.sampling_steps,
        "guidance_scale": args.guidance_scale,
        "guidance_energy": args.guidance_energy,
        "elapsed_seconds": elapsed,
        "diagnostics": {
            name: float(value) for name, value in diagnostics.items()
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure label-free routing proxies for v45 routing-training arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from floorflow.chip_flow.batched import (
    ROUTING_PROXY_REFERENCE,
    batch_routing_proxy_loss,
    batch_routing_proxy_terms,
)
from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from scripts.evaluate_chip_refiner import aggregate_physics, load_case_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, metavar="ARM=PATH")
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev",
    )
    parser.add_argument(
        "--case-indices",
        type=Path,
        default=ROOT / "outputs/reports/chip_dev_screen_v45.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_routing_proxy_screen.json",
    )
    return parser.parse_args()


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        arm, separator, path = value.partition("=")
        if not separator or not arm or not path:
            raise ValueError(f"expected ARM=PATH, got {value}")
        result[arm] = Path(path)
    return result


def measure(
    checkpoint: Path,
    store: PackedCaseStore,
    requested: set[int],
    device: torch.device,
    batch_size: int,
    nfe: int,
    seed: int,
) -> dict[str, float]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = load_chip_checkpoint(payload, device, use_ema=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    terms = {
        "pin_concentration_peak": [],
        "route_demand_peak": [],
        "occupancy_peak": [],
        "spacing_risk": [],
        "normalized_composite": [],
    }
    for count in sorted(store.by_block_count):
        indices = [
            int(index)
            for index in store.indices(count)
            if int(index) in requested
        ]
        for start in range(0, len(indices), batch_size):
            batch = store.batch(indices[start : start + batch_size], device=device)
            generator.manual_seed(seed)
            latent, _ = sample_for_batch(
                model,
                batch,
                sampling_steps=nfe,
                generator=generator,
            )
            boxes, _ = aggregate_physics(latent, batch)
            values = batch_routing_proxy_terms(
                boxes,
                batch["areas"],
                batch["b2b"],
                batch["p2b"],
                batch["pins"],
                batch["n_b2b"],
                batch["n_p2b"],
                batch["n_pins"],
            )
            for name, collected in terms.items():
                if name == "normalized_composite":
                    continue
                collected.extend(
                    float(value)
                    for value in values[name].flatten()
                )
            terms["normalized_composite"].extend(
                float(value)
                for value in batch_routing_proxy_loss(
                    boxes,
                    batch["areas"],
                    batch["b2b"],
                    batch["p2b"],
                    batch["pins"],
                    batch["n_b2b"],
                    batch["n_p2b"],
                    batch["n_pins"],
                ).flatten()
            )
    means = {
        name: sum(values) / len(values) if values else 0.0
        for name, values in terms.items()
    }
    means["composite"] = (
        means["pin_concentration_peak"]
        + 0.5 * means["route_demand_peak"]
        + 0.25 * means["occupancy_peak"]
    )
    return means


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    requested = load_case_indices(args.case_indices)
    if requested is None:
        raise ValueError("a frozen case-index manifest is required")
    device = torch.device(args.device)
    store = PackedCaseStore(args.cache)
    arms = {
        arm: measure(
            path,
            store,
            requested,
            device,
            args.batch_size,
            args.nfe,
            args.seed,
        )
        for arm, path in checkpoints.items()
    }
    control = arms.get("control")
    relative = {}
    if control is not None:
        relative = {
            arm: {
                key: value / control[key] - 1.0
                for key, value in metrics.items()
                if control[key] != 0.0
            }
            for arm, metrics in arms.items()
            if arm != "control"
        }
    report = {
        "version": "chip-v45-routing-proxy-screen-v2",
        "status": "complete",
        "scope": (
            "Dimensionless NFE4 boxes on the frozen 32-case dev screen; "
            "no legalizer or target labels."
        ),
        "reference": ROUTING_PROXY_REFERENCE,
        "seed": args.seed,
        "arms": arms,
        "relative_to_control": relative,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arms": arms, "relative_to_control": relative}, indent=2))


if __name__ == "__main__":
    main()

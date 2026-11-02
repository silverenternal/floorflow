#!/usr/bin/env python3
"""Collect a seed-19 refiner-step curve for graph-token flow and its baseline."""

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
from floorflow.chip_flow.flow_adapter import sample_for_batch
from scripts.evaluate_chip_official import load_model
from scripts.evaluate_chip_refiner import (
    aggregate_physics,
    make_instances,
    score_boxes,
)

ROOT_OUTPUTS = ROOT / "outputs"
STEPS = (0, 50, 100, 200, 400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT_OUTPUTS / "cache/chip_official_test_100_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_OUTPUTS / "reports/chip_token_flow_refiner_curve_v1.json",
    )
    return parser.parse_args()


def run_arm(model, store: PackedCaseStore, arm: str, args) -> list[dict]:
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(19)
    rows = []
    with torch.inference_mode():
        for count in sorted(store.by_block_count):
            indices = store.indices(count)
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                batch = store.batch(batch_indices, device=device)
                batch["_indices"] = torch.as_tensor(
                    batch_indices,
                    device=device,
                )
                generator.manual_seed(19)
                latent, _ = sample_for_batch(
                    model,
                    batch,
                    sampling_steps=4,
                    generator=generator,
                )
                refine_case = build_refine_case(
                    make_instances(batch),
                    1,
                    device,
                )
                snapshots = refine_closed(
                    latent,
                    refine_case,
                    PAPER_WEIGHTS,
                    400,
                    1.0e-3,
                    "linear",
                    snapshots=(0, 50, 100, 200),
                    chunk=5,
                    compile_loop=False,
                )
                for step in STEPS:
                    boxes, _ = aggregate_physics(snapshots[step], batch)
                    scored = score_boxes(boxes, batch, None)
                    for row in scored:
                        row["arm"] = arm
                        row["seed"] = 19
                        row["nfe"] = 4
                        row["refiner_steps"] = step
                    rows.extend(scored)
    return rows


def main() -> None:
    args = parse_args()
    store = PackedCaseStore(args.cache)
    rows = []
    started = time.perf_counter()
    for arm, root in (
        (
            "candidate",
            ROOT_OUTPUTS / "chip_trinity/rescue_20k/trinity_token_seed19_20k",
        ),
        (
            "baseline",
            ROOT_OUTPUTS / "chip_final/screen_20k/graph_velocity_seed19_20k_final",
        ),
    ):
        checkpoint = torch.load(
            root / "checkpoints/final.pt",
            map_location=args.device,
            weights_only=False,
        )
        model = load_model(
            checkpoint,
            argparse.Namespace(use_ema=False, device=args.device),
        )
        rows.extend(run_arm(model, store, arm, args))
        print(json.dumps({"arm": arm, "rows": len(rows)}), flush=True)
    report = {
        "status": "complete" if len(rows) == 2 * len(STEPS) * len(store) else "incomplete",
        "scope": "seed-19 matched NFE4 refiner-budget curve",
        "steps": list(STEPS),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "rows"}))


if __name__ == "__main__":
    main()

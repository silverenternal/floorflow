#!/usr/bin/env python3
"""Measure whether the shared refiner improves the supervised target itself."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import PAPER_WEIGHTS, refine_closed

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.model import decode_boxes_batch
from scripts.evaluate_chip_official import score_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument("--cases-per-count", type=int, default=8)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 5, 20, 100, 400])
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_refiner_target_probe_v1.json",
    )
    return parser.parse_args()


def floorplan_instance(
    payload: dict,
):
    from trinity.floorplan.types import FloorplanInstance

    count = int(payload["blocks"])
    constraints = payload["constraints"]
    boxes = payload["gt_boxes"]
    targets = np.full((count, 4), -1.0, dtype=np.float32)
    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    targets[fixed, 2:] = boxes[fixed, 2:]
    targets[preplaced] = boxes[preplaced]
    return FloorplanInstance(
        block_count=count,
        area_targets=payload["areas"],
        constraints=constraints,
        b2b=payload["b2b"],
        p2b=payload["p2b"],
        pins_pos=payload["pins"],
        target_positions=targets,
        gt_positions=boxes,
    )


def payload(batch: dict, row: int) -> dict:
    count = int(batch["n"])
    return {
        "index": row,
        "blocks": count,
        "boxes": batch["target_latent"][row].cpu().numpy().copy(),
        "gt_boxes": batch["boxes"][row, :count].cpu().numpy().copy(),
        "areas": batch["areas"][row, :count].cpu().numpy().copy(),
        "constraints": batch["constraints"][row, :count].cpu().numpy().copy(),
        "b2b": batch["b2b"][row, : int(batch["n_b2b"][row])]
        .cpu()
        .numpy()
        .copy(),
        "p2b": batch["p2b"][row, : int(batch["n_p2b"][row])]
        .cpu()
        .numpy()
        .copy(),
        "pins": batch["pins"][row, : int(batch["n_pins"][row])]
        .cpu()
        .numpy()
        .copy(),
        "latent_mse": 0.0,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    store = PackedCaseStore(args.cache)
    collected = {
        steps: {
            "raw": [],
            "legalized": [],
            "displacement": [],
        }
        for steps in args.steps
    }
    for count in sorted(store.by_block_count):
        indices = store.indices(count)[: args.cases_per_count]
        batch = store.batch(indices, device=device)
        rows = [payload(batch, index) for index in range(len(indices))]
        instances = [floorplan_instance(row) for row in rows]
        refine_case = build_refine_case(instances, 1, device)
        latent = batch["target_latent"].float()
        initial_boxes = batch["boxes"].float()
        for steps in args.steps:
            refined = (
                latent
                if steps == 0
                else refine_closed(
                    latent,
                    refine_case,
                    PAPER_WEIGHTS,
                    steps,
                    1.0e-3,
                    "linear",
                    chunk=5,
                    compile_loop=False,
                )[steps]
            )
            boxes = decode_boxes_batch(refined, batch["areas"].float())
            for index, row in enumerate(rows):
                scored = score_case({**row, "boxes": boxes[index].cpu().numpy(), "legalize": True})
                collected[steps]["raw"].append(scored["raw_hard_cost"])
                collected[steps]["legalized"].append(
                    scored["legalized_hard_cost"]
                )
                collected[steps]["displacement"].append(
                    float(
                        (boxes[index] - initial_boxes[index])
                        .abs()
                        .mean()
                    )
                )
    results = {
        str(steps): {
            "raw_hard_cost": float(np.mean(values["raw"])),
            "legalized_hard_cost": float(np.mean(values["legalized"])),
            "mean_abs_box_displacement": float(
                np.mean(values["displacement"])
            ),
        }
        for steps, values in collected.items()
    }
    baseline = results["0"]
    for values in results.values():
        values["legalized_relative_delta"] = (
            values["legalized_hard_cost"]
            - baseline["legalized_hard_cost"]
        ) / baseline["legalized_hard_cost"]
    best_step, best = min(
        results.items(),
        key=lambda item: item[1]["legalized_hard_cost"],
    )
    checks = {
        "target_refinement_improves_legalized_cost_1pct": (
            best["legalized_relative_delta"] < -0.01
        ),
        "best_step_moves_target": best["mean_abs_box_displacement"] > 0.0,
        "best_step_not_zero": best_step != "0",
    }
    report = {
        "status": "target_refinement_has_headroom" if all(checks.values()) else "reject_target_refinement",
        "scope": "fixed training-target layouts through shared closed-form refiner; no model quality claim",
        "cases": sum(args.cases_per_count for _ in store.by_block_count),
        "steps": args.steps,
        "results": results,
        "best_step": best_step,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

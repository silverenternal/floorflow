#!/usr/bin/env python3
"""Evaluate a released Trinity checkpoint on official FloorSet validation cases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party/Trinity/src"))

from trinity import REFINER, SAMPLER, build
from trinity.floorplan.data import load_validation_case
from trinity.floorplan.legalize import pack_lp
from trinity.floorplan.scoring import validate
from trinity.floorplan.scoring.cost import score_full_fast
from trinity.solver import DiffusionPlacer

from floorflow.chip_flow.refiner_config import parse_refiner_weight_overrides
from floorflow.chip_flow.reproducibility import (
    enable_deterministic_execution,
    stable_batch_seed,
)
from floorflow.chip_flow.score_diagnostics import score_diagnostics
from floorflow.chip_flow.trinity_release import ReleaseComponents, load_release


def make_placer(
    release: ReleaseComponents,
    args: argparse.Namespace,
    refiner: dict | None,
) -> DiffusionPlacer:
    sampler = build(
        {
            "name": release.config["sampler"]["name"],
            "num_steps": args.sampler_steps,
            "projections": [] if args.free_sampling else None,
        },
        SAMPLER,
        framing=release.framing,
    )
    return DiffusionPlacer(
        release.backbone,
        sampler,
        build(refiner, REFINER),
        param=release.param,
        samples=args.samples,
        scorer=args.scorer,
        device=args.device,
        legalize_workers=args.legalize_workers,
        graph_pe=release.graph_pe,
        cond_options=release.cond_options,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "third_party/Trinity/checkpoints/d192",
    )
    parser.add_argument("--blocks", type=int, nargs="+", default=list(range(21, 121)))
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sampler-steps", type=int, default=4)
    parser.add_argument("--refiner-steps", type=int, default=400)
    parser.add_argument("--refiner-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--refiner-schedule",
        choices=("linear", "cosine", "constant"),
        default="linear",
    )
    parser.add_argument("--refiner-weight", action="append", default=[])
    parser.add_argument(
        "--scorer",
        choices=("full", "full_fast"),
        default="full_fast",
    )
    parser.add_argument("--free-sampling", action="store_true")
    parser.add_argument("--legalize-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--legalizer-time-limit-ms",
        type=int,
        default=2000,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.deterministic:
        enable_deterministic_execution()
    previous_lp_timeout = pack_lp.TIME_LIMIT_MS
    pack_lp.TIME_LIMIT_MS = args.legalizer_time_limit_ms
    release = load_release(args.checkpoint, args.device)
    refiner_weights = parse_refiner_weight_overrides(args.refiner_weight)
    refiner = (
        None
        if args.refiner_steps <= 0
        else {
            "name": "closed",
            "steps": args.refiner_steps,
            "lr": args.refiner_lr,
            "lr_schedule": args.refiner_schedule,
            "weights": refiner_weights,
            "compile_loop": False,
        }
    )
    placer = make_placer(release, args, refiner)
    rows = []
    try:
        for blocks in args.blocks:
            started = time.perf_counter()
            if args.seed is not None:
                torch.manual_seed(
                    stable_batch_seed(args.seed, [int(blocks) - 21])
                )
            instance = load_validation_case(int(blocks))
            placement = placer.solve(instance)
            score = (
                score_full_fast(placement)
                if args.scorer == "full_fast"
                else validate(placement, "full").score
            )
            row = {
                "index": int(blocks) - 21,
                "blocks": int(blocks),
                "feasible": bool(score.feasible),
                "cost": float(score.cost),
                "hpwl": float(score.hpwl),
                "hpwl_gap": float(score.hpwl_gap),
                "area_gap": float(score.area_gap),
                "v_rel": float(score.v_rel),
                "overlap_pairs": len(score.feasibility.overlap_pairs),
                "grouping": int(score.soft.grouping) if score.soft else 0,
                "mib": int(score.soft.mib) if score.soft else 0,
                "boundary": int(score.soft.boundary) if score.soft else 0,
                "seconds": time.perf_counter() - started,
            }
            row.update(
                score_diagnostics(
                    "legalized",
                    score,
                    np.asarray(placement.xywh),
                    np.asarray(instance.gt_positions),
                    np.asarray(instance.b2b),
                    np.asarray(instance.p2b),
                    np.asarray(instance.pins_pos),
                )
            )
            rows.append(row)
            print(json.dumps(rows[-1]), flush=True)
    finally:
        pack_lp.TIME_LIMIT_MS = previous_lp_timeout
    costs = [row["cost"] for row in rows]
    block_counts = [row["blocks"] for row in rows]
    report = {
        "checkpoint": str(args.checkpoint),
        "protocol": {
            "samples_per_case": args.samples,
            "sampler_steps": args.sampler_steps,
            "refiner": refiner,
            "legalizer": "scale_pack default route",
            "scorer": args.scorer,
            "free_sampling": bool(args.free_sampling),
            "legalize_workers": args.legalize_workers,
            "best_of_n_used": False,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "legalizer_time_limit_ms": args.legalizer_time_limit_ms,
            "legalizer_glop_threads": 1,
            "device": args.device,
        },
        "cases": len(rows),
        "rows": rows,
        "mean_cost": float(np.mean(costs)),
        "cost_exp_weighted": float(
            np.sum(np.asarray(costs) * np.exp(np.asarray(block_counts) / 12.0))
            / np.sum(np.exp(np.asarray(block_counts) / 12.0))
        ),
        "cost_blockcount_weighted": float(
            np.sum(np.asarray(costs) * np.asarray(block_counts))
            / np.sum(block_counts)
        ),
        "mean_v_rel": float(np.mean([row["v_rel"] for row in rows])),
        "mean_hpwl_gap": float(np.mean([row["hpwl_gap"] for row in rows])),
        "mean_area_gap": float(np.mean([row["area_gap"] for row in rows])),
        "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
        "feasible_fraction": sum(row["feasible"] for row in rows) / len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "cases",
                    "mean_cost",
                    "cost_exp_weighted",
                    "cost_blockcount_weighted",
                    "mean_v_rel",
                    "feasible_fraction",
                    "mean_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

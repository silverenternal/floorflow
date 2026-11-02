#!/usr/bin/env python3
"""Collect label-free observables for bounded adaptive refiner stopping."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed

from floorflow.chip_flow.batched import batch_physics_terms
from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.refiner_config import parse_refiner_weight_overrides
from floorflow.chip_flow.reproducibility import (
    enable_deterministic_execution,
    stable_batch_seed,
)
from floorflow.chip_physics import group_penalty
from scripts.evaluate_chip_refiner import (
    load_case_indices,
    make_instances,
    score_boxes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case-indices", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[400, 800, 1600, 3200])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--legalizer-time-limit-ms", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--refiner-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--refiner-schedule",
        choices=("linear", "cosine", "constant"),
        default="linear",
    )
    parser.add_argument("--refiner-weight", action="append", default=[])
    parser.add_argument("--refiner-shape", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refiner-optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--refiner-chunk", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_sota_adaptive_snapshots.json",
    )
    return parser.parse_args()


def signed_rms(delta: torch.Tensor) -> float:
    return float(torch.sqrt(delta.float().square().mean()))


def case_physics(boxes: torch.Tensor, batch: dict) -> list[dict[str, float]]:
    terms = batch_physics_terms(
        boxes,
        batch["areas"],
        batch["constraints"],
        batch["b2b"],
        batch["p2b"],
        batch["pins"],
        batch["n_b2b"],
        batch["n_p2b"],
        batch["n_pins"],
    )
    group = torch.stack(
        [
            group_penalty(
                boxes[index],
                batch["constraints"][index, :, 3],
                batch["areas"][index],
            )
            for index in range(boxes.shape[0])
        ]
    )
    return [
        {
            **{
                name: float(value[index])
                for name, value in terms.items()
            },
            "group": float(group[index]),
        }
        for index in range(boxes.shape[0])
    ]


def observable_row(
    *,
    case_index: int,
    steps: int,
    snapshots: list[int],
    initial: torch.Tensor,
    previous: torch.Tensor,
    current: torch.Tensor,
    physics: dict[str, float],
) -> dict:
    step_index = snapshots.index(steps)
    return {
        "index": case_index,
        "refiner_steps": steps,
        "refiner_step_delta": steps - (snapshots[step_index - 1] if step_index else 0),
        "trust_region_rms": signed_rms(current - initial),
        "incremental_rms": signed_rms(current - previous),
        **physics,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    snapshots = sorted(set(args.snapshots))
    if not snapshots or snapshots[0] <= 0:
        raise ValueError("snapshots must contain positive increasing steps")
    if args.deterministic:
        enable_deterministic_execution()
    device = torch.device(args.device)
    weights = parse_refiner_weight_overrides(args.refiner_weight)
    store = PackedCaseStore(args.cache)
    requested = load_case_indices(args.case_indices)
    available = {
        int(index)
        for count in store.by_block_count
        for index in store.indices(count)
    }
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"case manifest indices absent from cache: {missing}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = load_chip_checkpoint(checkpoint, args.device, use_ema=args.use_ema)
    executor = (
        ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            mp_context=multiprocessing.get_context("spawn"),
        )
        if args.workers > 1
        else None
    )
    rows = []
    legalized = []
    started = time.perf_counter()
    try:
        for count in sorted(store.by_block_count):
            indices = [
                int(index)
                for index in store.indices(count)
                if int(index) in requested
            ]
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                batch = store.batch(batch_indices, device=device)
                batch["_indices"] = torch.as_tensor(
                    batch_indices,
                    device=device,
                )
                generator = torch.Generator(device=device).manual_seed(
                    stable_batch_seed(args.seed, batch_indices)
                )
                latent, _ = sample_for_batch(
                    model,
                    batch,
                    sampling_steps=args.nfe,
                    generator=generator,
                )
                case = build_refine_case(make_instances(batch), 1, device)
                refined = refine_closed(
                    latent,
                    case,
                    weights,
                    max(snapshots),
                    args.refiner_lr,
                    args.refiner_schedule,
                    snapshots=(0, *snapshots),
                    chunk=args.refiner_chunk,
                    compile_loop=False,
                    shape=args.refiner_shape,
                    optimizer=args.refiner_optimizer,
                )
                previous = refined[0]
                for steps in snapshots:
                    current = refined[steps]
                    latent_boxes = project_conditioned_latent(
                        current,
                        batch["target_latent"],
                        batch["constraints"],
                    )
                    boxes = decode_boxes_batch(latent_boxes, batch["areas"])
                    physics_rows = case_physics(boxes, batch)
                    scored = score_boxes(
                        boxes,
                        batch,
                        executor,
                        args.legalizer_time_limit_ms,
                    )
                    for batch_row, case_index in enumerate(batch_indices):
                        rows.append(
                            observable_row(
                                case_index=case_index,
                                steps=steps,
                                snapshots=snapshots,
                                initial=refined[0][batch_row],
                                previous=previous[batch_row],
                                current=current[batch_row],
                                physics=physics_rows[batch_row],
                            )
                        )
                        legalized.append(
                            {
                                "index": case_index,
                                "nfe": args.nfe,
                                "refiner_steps": steps,
                                **scored[batch_row],
                            }
                        )
                    previous = current
    finally:
        if executor is not None:
            executor.shutdown()
    expected_rows = len(requested) * len(snapshots)
    if len(rows) != expected_rows or len(legalized) != expected_rows:
        raise RuntimeError(
            f"incomplete snapshot collection: expected {expected_rows}, "
            f"found {len(rows)} observables and {len(legalized)} legalized rows"
        )
    report = {
        "version": "chip-sota-adaptive-snapshots-v1",
        "status": "complete",
        "scope": "label-free observables plus held-out metric analysis",
        "checkpoint": str(args.checkpoint),
        "case_manifest": str(args.case_indices),
        "cache": str(args.cache),
        "device": str(device),
        "nfe": args.nfe,
        "snapshots": snapshots,
        "seed": args.seed,
        "use_ema": args.use_ema,
        "deterministic": args.deterministic,
        "refiner_weights": weights,
        "compute_accounting": {
            "refiner_steps_performed": max(snapshots),
            "observed_snapshot_count": len(snapshots),
            "generated_draws": len(requested),
            "legalized_candidates": len(legalized),
            "wall_clock_seconds": time.perf_counter() - started,
        },
        "target_used_for_runtime_stop": False,
        "target_labels_recorded_for_offline_calibration": True,
        "case_identity_used_for_stop": False,
        "rows": rows,
        "legalized": legalized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"rows", "legalized"}}))


if __name__ == "__main__":
    main()

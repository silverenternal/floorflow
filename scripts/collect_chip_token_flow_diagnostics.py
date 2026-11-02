#!/usr/bin/env python3
"""Collect matched NFE curves and per-case score components for graph-token flow."""

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
from trinity.sampling.refine_closed import PAPER_WEIGHTS, refine_closed

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.flow_adapter import sample_for_batch
from scripts.evaluate_chip_official import load_model
from scripts.evaluate_chip_refiner import (
    aggregate_physics,
    make_instances,
    score_boxes,
)

SEEDS = (19, 20, 21)
NFES = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=ROOT / "outputs/chip_trinity/rescue_20k",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=ROOT / "outputs/chip_final/screen_20k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_component_rows_v1.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refiner-steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--nfes", type=int, nargs="+", default=list(NFES))
    parser.add_argument("--blocks", type=int, nargs="+", default=None)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("candidate", "baseline"),
        default=["candidate", "baseline"],
    )
    return parser.parse_args()


def checkpoint_path(root: Path, arm: str, seed: int) -> Path:
    if arm == "candidate":
        arm_dir = root / f"trinity_token_seed{seed}_20k"
    else:
        arm_dir = root / f"graph_velocity_seed{seed}_20k_final"
    return arm_dir / "checkpoints/final.pt"


def add_complexity(rows: list[dict], batch: dict, metadata: dict) -> None:
    constraints = batch["constraints"].detach().cpu()
    for index, row in enumerate(rows):
        current = constraints[index]
        row.update(
            {
                **metadata,
                "n_b2b": int(batch["n_b2b"][index]),
                "n_p2b": int(batch["n_p2b"][index]),
                "n_pins": int(batch["n_pins"][index]),
                "boundary_blocks": int(current[:, 4].ne(0).sum()),
                "cluster_groups": int(
                    torch.unique(current[current[:, 3] > 0, 3]).numel()
                ),
                "mib_groups": int(
                    torch.unique(current[current[:, 2] > 0, 2]).numel()
                ),
            }
        )


def collect_arm(
    model,
    store: PackedCaseStore,
    *,
    arm: str,
    seed: int,
    args: argparse.Namespace,
    executor: ProcessPoolExecutor | None,
) -> list[dict]:
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = []
    with torch.inference_mode():
        for count in sorted(store.by_block_count):
            if args.blocks is not None and count not in args.blocks:
                continue
            indices = store.indices(count)
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                batch = store.batch(batch_indices, device=device)
                batch["_indices"] = torch.as_tensor(
                    batch_indices,
                    device=device,
                )
                metadata = {
                    "seed": seed,
                    "arm": arm,
                    "blocks": count,
                }
                for nfe in args.nfes:
                    generator.manual_seed(seed)
                    latent, _ = sample_for_batch(
                        model,
                        batch,
                        sampling_steps=nfe,
                        generator=generator,
                    )
                    boxes, _ = aggregate_physics(latent, batch)
                    scored = score_boxes(boxes, batch, executor)
                    add_complexity(
                        scored,
                        batch,
                        {**metadata, "nfe": nfe, "stage": "raw"},
                    )
                    rows.extend(scored)
                refine_case = build_refine_case(
                    make_instances(batch),
                    1,
                    device,
                )
                generator.manual_seed(seed)
                latent, _ = sample_for_batch(
                    model,
                    batch,
                    sampling_steps=4,
                    generator=generator,
                )
                refined = refine_closed(
                    latent,
                    refine_case,
                    PAPER_WEIGHTS,
                    args.refiner_steps,
                    1.0e-3,
                    "linear",
                    chunk=5,
                    compile_loop=False,
                )[args.refiner_steps]
                boxes, _ = aggregate_physics(refined, batch)
                scored = score_boxes(boxes, batch, executor)
                add_complexity(
                    scored,
                    batch,
                    {
                        **metadata,
                        "nfe": 4,
                        "stage": f"refine{args.refiner_steps}",
                    },
                )
                rows.extend(scored)
    return rows


def main() -> None:
    args = parse_args()
    store = PackedCaseStore(args.cache)
    executor = (
        ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            mp_context=multiprocessing.get_context("spawn"),
        )
        if args.workers > 1
        else None
    )
    rows = []
    started = time.perf_counter()
    try:
        for seed in args.seeds:
            for arm, root in (
                ("candidate", args.candidate_root),
                ("baseline", args.baseline_root),
            ):
                if arm not in args.arms:
                    continue
                checkpoint = torch.load(
                    checkpoint_path(root, arm, seed),
                    map_location=args.device,
                    weights_only=False,
                )
                model_arguments = argparse.Namespace(use_ema=False, device=args.device)
                model = load_model(checkpoint, model_arguments)
                rows.extend(
                    collect_arm(
                        model,
                        store,
                        arm=arm,
                        seed=seed,
                        args=args,
                        executor=executor,
                    )
                )
                print(
                    json.dumps(
                        {"seed": seed, "arm": arm, "rows": len(rows)},
                    ),
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    selected_cases = sum(
        len(store.indices(count))
        for count in store.by_block_count
        if args.blocks is None or count in args.blocks
    )
    expected = (
        len(args.seeds)
        * len(args.arms)
        * (len(args.nfes) + 1)
        * selected_cases
    )
    report = {
        "status": "complete" if len(rows) == expected else "incomplete",
        "scope": "matched official per-case NFE and score-component diagnostics",
        "cache": str(args.cache),
        "seeds": args.seeds,
        "nfes": args.nfes,
        "refiner_steps": args.refiner_steps,
        "expected_rows": expected,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "rows"}))


if __name__ == "__main__":
    main()

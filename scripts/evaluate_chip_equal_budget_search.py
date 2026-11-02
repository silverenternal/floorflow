#!/usr/bin/env python3
"""Evaluate label-free equal-refiner-budget proposal search for chip flow."""

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
from floorflow.chip_flow.evaluation_accumulators import physics_accumulator
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.refiner_config import parse_refiner_weight_overrides
from floorflow.chip_flow.reproducibility import enable_deterministic_execution
from floorflow.chip_physics import group_penalty
from scripts.evaluate_chip_refiner import (
    load_case_indices,
    make_instances,
    score_boxes,
    score_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--proposal-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--total-refiner-steps", type=int, default=1600)
    parser.add_argument("--workers", type=int, default=4)
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
        default=ROOT / "outputs/reports/chip_v45_inference_search_screen.json",
    )
    return parser.parse_args()


def reranker_energy(
    boxes: torch.Tensor,
    batch: dict,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    return (
        weights["overlap"] * terms["overlap"]
        + weights["group"] * group
        + weights["mib"] * terms["mib"]
        + weights["boundary"] * terms["boundary"]
        + weights["wl"] * terms["wl"]
        + weights["area"] * terms["area"]
    ), {
        **{name: value.detach() for name, value in terms.items() if name != "total"},
        "group": group.detach(),
    }


def select_proposals(
    proposals: list[torch.Tensor],
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = scores.argmin(dim=0)
    stacked = torch.stack(proposals, dim=0)
    batch_rows = torch.arange(selected.numel(), device=selected.device)
    return stacked[selected, batch_rows], selected


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.deterministic:
        enable_deterministic_execution()
    if any(count <= 0 for count in args.proposal_counts):
        raise ValueError("proposal counts must be positive")
    if any(args.total_refiner_steps % count for count in args.proposal_counts):
        raise ValueError("total refiner steps must divide every proposal count")

    device = torch.device(args.device)
    weights = parse_refiner_weight_overrides(args.refiner_weight)
    store = PackedCaseStore(args.cache)
    requested = load_case_indices(args.case_indices)
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
    results = {}
    all_rows = {}
    try:
        for count in args.proposal_counts:
            started = time.perf_counter()
            steps_each = args.total_refiner_steps // count
            collected = {
                "rows": [],
                "physics": physics_accumulator(),
                "selected": [],
            }
            for blocks in sorted(store.by_block_count):
                indices = [
                    int(index)
                    for index in store.indices(blocks)
                    if requested is None or int(index) in requested
                ]
                for start in range(0, len(indices), args.batch_size):
                    batch_indices = indices[start : start + args.batch_size]
                    batch = store.batch(batch_indices, device=device)
                    batch["_indices"] = torch.as_tensor(
                        batch_indices,
                        device=device,
                    )
                    refine_case = build_refine_case(
                        make_instances(batch),
                        1,
                        device,
                    )
                    proposals = []
                    proposal_scores = []
                    for proposal in range(count):
                        generator = torch.Generator(device=device).manual_seed(
                            args.seed + 1000 * proposal
                        )
                        latent, _ = sample_for_batch(
                            model,
                            batch,
                            sampling_steps=args.nfe,
                            generator=generator,
                        )
                        refined = refine_closed(
                            latent,
                            refine_case,
                            weights,
                            steps_each,
                            args.refiner_lr,
                            args.refiner_schedule,
                            chunk=args.refiner_chunk,
                            compile_loop=False,
                            shape=args.refiner_shape,
                            optimizer=args.refiner_optimizer,
                        )[steps_each]
                        projected = project_conditioned_latent(
                            refined.float(),
                            batch["target_latent"],
                            batch["constraints"],
                        )
                        boxes = decode_boxes_batch(projected, batch["areas"])
                        energy, _ = reranker_energy(boxes, batch, weights)
                        proposals.append(projected)
                        proposal_scores.append(energy)
                    energy = torch.stack(proposal_scores)
                    selected_latent, selected_proposal = select_proposals(
                        proposals,
                        energy,
                    )
                    boxes = decode_boxes_batch(selected_latent, batch["areas"])
                    _, physics = reranker_energy(boxes, batch, weights)
                    for name, value in physics.items():
                        collected["physics"][name] += float(value.sum())
                    rows = score_boxes(
                        boxes,
                        batch,
                        executor,
                        args.legalizer_time_limit_ms,
                    )
                    collected["rows"].extend(rows)
                    collected["selected"].extend(
                        {
                            "case_index": case_index,
                            "proposal": int(proposal),
                            "proposal_scores": scores.tolist(),
                        }
                        for case_index, proposal, scores in zip(
                            batch_indices,
                            selected_proposal.tolist(),
                            energy.transpose(0, 1),
                        )
                    )
            summary = score_summary(collected["rows"], collected["physics"])
            summary.update(
                {
                    "proposals": count,
                    "refiner_steps_per_proposal": steps_each,
                    "total_refiner_steps": args.total_refiner_steps,
                    "total_nfe_evaluations": count * args.nfe,
                    "wall_clock_seconds": time.perf_counter() - started,
                    "selection_counts": {
                        str(index): sum(
                            row["proposal"] == index for row in collected["selected"]
                        )
                        for index in range(count)
                    },
                }
            )
            key = f"nfe{args.nfe}_proposals{count}_total{args.total_refiner_steps}"
            results[key] = summary
            all_rows[key] = collected["selected"]
    finally:
        if executor is not None:
            executor.shutdown()

    report = {
        "version": "chip-v45-label-free-search-v1",
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "case_indices": str(args.case_indices),
        "device": str(device),
        "seed": args.seed,
        "use_ema": args.use_ema,
        "nfe": args.nfe,
        "total_refiner_steps": args.total_refiner_steps,
        "selection": {
            "official_metrics_used": False,
            "target_derived_selection": False,
            "score": "label-free weighted structural physics",
            "weights": weights,
        },
        "results": results,
        "selected_proposals": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate cached chip predictions through Trinity's existing refiner."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.scoring.total import (
    total_blockcount,
    total_exp,
    total_mean,
)
from trinity.floorplan.types import FloorplanInstance
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import (
    refine_closed,
)

from floorflow.chip_flow.batched import batch_physics_terms
from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.evaluation_accumulators import physics_accumulator
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.legalizer_policy import LegalizerPolicy
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import (
    project_conditioned_latent,
)
from floorflow.chip_flow.refiner_config import (
    configure_chip_projector,
    parse_refiner_weight_overrides,
    project_after_refiner_batch,
    refiner_comparison_status,
)
from floorflow.chip_flow.reproducibility import enable_deterministic_execution, stable_batch_seed
from floorflow.chip_physics import group_penalty
from scripts.evaluate_chip_official import (
    case_payload,
    score_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ensemble-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nfe", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--refiner-steps", type=int, nargs="+", default=[0, 100, 400])
    parser.add_argument("--blocks", type=int, nargs="+", default=None)
    parser.add_argument(
        "--case-indices",
        type=Path,
        default=None,
        help="Optional JSON list or {'indices': [...]} frozen evaluation subset.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--legalizer-time-limit-ms",
        type=int,
        default=2000,
    )
    parser.add_argument("--legalizer-policy-json", type=Path)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--refiner-lr", type=float, default=1.0e-3)
    parser.add_argument("--refiner-schedule", choices=("linear", "cosine", "constant"), default="linear")
    parser.add_argument(
        "--refiner-schedule-json",
        type=Path,
        default=None,
        help="Optional AnySchedule JSON object; overrides --refiner-schedule.",
    )
    parser.add_argument(
        "--refiner-weight",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a closed-form constraint weight; may be repeated.",
    )
    parser.add_argument(
        "--refiner-shape",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--refiner-optimizer",
        choices=("adam", "sgd"),
        default="adam",
    )
    parser.add_argument("--refiner-compile-loop", action="store_true")
    parser.add_argument("--refiner-chunk", type=int, default=5)
    parser.add_argument("--projector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--projector-iterations", type=int, default=None)
    parser.add_argument("--projector-boundary-axes", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--projector-boundary-lock-corners",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--post-refiner-projector-iterations", type=int, default=0)
    parser.add_argument("--post-refiner-projector-strength", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--guidance-energy", choices=("routing", "composite"), default="routing")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_inference_refiner_comparison_v1.json",
    )
    return parser.parse_args()


def make_instances(batch: dict) -> list[FloorplanInstance]:
    instances = []
    for index in range(batch["features"].shape[0]):
        count = int(batch["n"])
        constraints = batch["constraints"][index, :count].cpu().numpy()
        boxes = batch["boxes"][index, :count].cpu().numpy()
        targets = np.full((count, 4), -1.0, dtype=np.float32)
        fixed = constraints[:, 0] != 0
        preplaced = constraints[:, 1] != 0
        targets[fixed, 2:] = boxes[fixed, 2:]
        targets[preplaced] = boxes[preplaced]
        instances.append(
            FloorplanInstance(
                block_count=count,
                area_targets=batch["areas"][index, :count].cpu().numpy(),
                constraints=constraints,
                b2b=batch["b2b"][
                    index, : int(batch["n_b2b"][index])
                ].cpu().numpy(),
                p2b=batch["p2b"][
                    index, : int(batch["n_p2b"][index])
                ].cpu().numpy(),
                pins_pos=batch["pins"][
                    index, : int(batch["n_pins"][index])
                ].cpu().numpy(),
                target_positions=targets,
                gt_positions=boxes,
            )
        )
    return instances


def load_case_indices(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("indices", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("case-index manifest must contain a non-empty list")
    indices = {int(value) for value in values}
    if len(indices) != len(values) or min(indices) < 0:
        raise ValueError("case-index manifest must contain unique non-negative indices")
    return indices


def aggregate_physics(
    latent: torch.Tensor,
    batch: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent = project_conditioned_latent(
        latent,
        batch["target_latent"],
        batch["constraints"],
    )
    boxes = decode_boxes_batch(latent, batch["areas"])
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
            for index in range(latent.shape[0])
        ]
    )
    values = {
        name: float(value.sum())
        for name, value in terms.items()
    }
    values["group"] = float(group.sum())
    return boxes, values


def score_boxes(
    boxes: torch.Tensor,
    batch: dict,
    executor: ProcessPoolExecutor | None,
    legalizer_time_limit_ms: int = 2000,
    legalizer_options: dict | None = None,
) -> list[dict]:
    payloads = [
        case_payload(
            batch,
            index,
            boxes[index],
            0.0,
            legalize=True,
            legalizer_time_limit_ms=legalizer_time_limit_ms,
            legalizer_options=legalizer_options,
            return_raw_boxes=True,
            return_legalized_boxes=True,
        )
        for index in range(boxes.shape[0])
    ]
    if executor is None:
        return [score_case(payload) for payload in payloads]
    return list(executor.map(score_case, payloads))


def score_summary(rows: list[dict], physics: dict[str, float]) -> dict:
    costs = [float(row["legalized_hard_cost"]) for row in rows]
    counts = [int(row["blocks"]) for row in rows]
    cases = len(rows)
    summary = {
        "cases": cases,
        "legalized_hard_cost": total_mean(costs, counts),
        "legalized_hard_cost_exp": total_exp(costs, counts),
        "legalized_hard_cost_blockcount": total_blockcount(costs, counts),
        "legalized_v_rel": float(
            np.mean([float(row["legalized_v_rel"]) for row in rows])
        ),
        "legalized_hpwl_gap": float(
            np.mean([float(row["legalized_hpwl_gap"]) for row in rows])
        ),
        "legalized_area_gap": float(
            np.mean([float(row["legalized_area_gap"]) for row in rows])
        ),
        "legalized_feasible_fraction": float(
            np.mean([bool(row["legalized_feasible"]) for row in rows])
        ),
        "raw_overlap": physics["overlap"] / cases,
        "raw_group": physics["group"] / cases,
        "raw_mib": physics["mib"] / cases,
        "raw_boundary": physics["boundary"] / cases,
        "raw_wl": physics["wl"] / cases,
        "raw_area": physics["area"] / cases,
    }
    for name in (
        "legalized_b2b_hpwl_gap",
        "legalized_p2b_hpwl_gap",
        "legalized_grouping",
        "legalized_mib",
        "legalized_boundary",
        "legalized_rho_mae",
    ):
        if name in rows[0]:
            summary[name] = float(np.mean([float(row[name]) for row in rows]))
    return summary


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    policy = (
        LegalizerPolicy(**json.loads(args.legalizer_policy_json.read_text()))
        if args.legalizer_policy_json is not None
        else None
    )
    if policy is not None:
        policy.validate()
    if args.deterministic:
        enable_deterministic_execution()
    device = torch.device(args.device)
    refiner_weights = parse_refiner_weight_overrides(args.refiner_weight)
    refiner_schedule: str | dict = args.refiner_schedule
    if args.refiner_schedule_json is not None:
        refiner_schedule = json.loads(
            args.refiner_schedule_json.read_text(encoding="utf-8")
        )
        if not isinstance(refiner_schedule, dict):
            raise ValueError("refiner schedule JSON must contain an object")
    store = PackedCaseStore(args.cache)
    requested_indices = load_case_indices(args.case_indices)
    available_indices = {
        int(index)
        for count in store.by_block_count
        for index in store.indices(count)
    }
    missing_indices = sorted((requested_indices or set()) - available_indices)
    if missing_indices:
        raise ValueError(f"case manifest indices absent from cache: {missing_indices}")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = load_chip_checkpoint(
        checkpoint,
        args.device,
        use_ema=args.use_ema,
    )
    model = configure_chip_projector(
        model,
        enabled=args.projector,
        iterations=args.projector_iterations,
        boundary_axes=args.projector_boundary_axes,
        boundary_lock_corners=args.projector_boundary_lock_corners,
    )
    ensemble_model = (
        load_chip_checkpoint(
            torch.load(
                args.ensemble_checkpoint,
                map_location=device,
                weights_only=False,
            ),
            args.device,
            use_ema=args.use_ema,
        )
        if args.ensemble_checkpoint is not None
        else None
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    executor = (
        ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            mp_context=multiprocessing.get_context("spawn"),
        )
        if args.workers > 1
        else None
    )
    results = {}
    case_rows = {}
    try:
        for nfe in args.nfe:
            nfe_started = time.perf_counter()
            collected = {
                steps: {
                    "rows": [],
                    "physics": physics_accumulator(),
                    "updates": {
                        "l1": 0.0,
                        "l2": 0.0,
                        "squared": 0.0,
                        "coordinates": 0.0,
                    },
                }
                for steps in args.refiner_steps
            }
            sample_seconds = refine_seconds = decode_seconds = score_seconds = 0.0
            for count in sorted(store.by_block_count):
                if args.blocks is not None and count not in args.blocks:
                    continue
                indices = [
                    index
                    for index in store.indices(count)
                    if requested_indices is None or int(index) in requested_indices
                ]
                for start in range(0, len(indices), args.batch_size):
                    batch_indices = indices[start : start + args.batch_size]
                    batch = store.batch(batch_indices, device=device)
                    batch["_indices"] = torch.as_tensor(
                        batch_indices,
                        device=device,
                    )
                    sample_started = time.perf_counter()
                    generator.manual_seed(stable_batch_seed(args.seed, batch_indices))
                    latent, _ = sample_for_batch(
                        model,
                        batch,
                        sampling_steps=nfe,
                        generator=generator,
                        guidance_scale=args.guidance_scale,
                        guidance_energy=args.guidance_energy,
                    )
                    if ensemble_model is not None:
                        generator.manual_seed(stable_batch_seed(args.seed, batch_indices))
                        ensemble_latent, _ = sample_for_batch(
                            ensemble_model,
                            batch,
                            sampling_steps=nfe,
                            generator=generator,
                            guidance_scale=args.guidance_scale,
                            guidance_energy=args.guidance_energy,
                        )
                        latent = 0.5 * (latent + ensemble_latent)
                    sample_seconds += time.perf_counter() - sample_started
                    refine_case = build_refine_case(
                        make_instances(batch),
                        1,
                        device,
                    )
                    for steps in args.refiner_steps:
                        refine_started = time.perf_counter()
                        refined = (
                            latent
                            if steps == 0
                            else refine_closed(
                                latent,
                                refine_case,
                                refiner_weights,
                                steps,
                                args.refiner_lr,
                                copy.deepcopy(refiner_schedule),
                                chunk=args.refiner_chunk,
                                compile_loop=args.refiner_compile_loop,
                                shape=args.refiner_shape,
                                optimizer=args.refiner_optimizer,
                            )[steps]
                        )
                        if args.post_refiner_projector_iterations > 0:
                            refined = project_after_refiner_batch(
                                refined,
                                batch,
                                model,
                                iterations=args.post_refiner_projector_iterations,
                                strength=args.post_refiner_projector_strength,
                            )
                        latent_delta = refined.float() - latent.float()
                        collected[steps]["updates"]["l1"] += float(
                            latent_delta.abs().sum()
                        )
                        collected[steps]["updates"]["l2"] += float(
                            torch.linalg.vector_norm(
                                latent_delta,
                                dim=(-1, -2),
                            ).sum()
                        )
                        collected[steps]["updates"]["squared"] += float(
                            latent_delta.square().sum()
                        )
                        collected[steps]["updates"]["coordinates"] += int(
                            latent_delta.numel()
                        )
                        refine_seconds += time.perf_counter() - refine_started
                        decode_started = time.perf_counter()
                        boxes, physics = aggregate_physics(refined, batch)
                        decode_seconds += time.perf_counter() - decode_started
                        score_started = time.perf_counter()
                        rows = score_boxes(
                            boxes,
                            batch,
                            executor,
                            args.legalizer_time_limit_ms,
                            policy.kwargs() if policy is not None else None,
                        )
                        score_seconds += time.perf_counter() - score_started
                        collected[steps]["rows"].extend(rows)
                        for name, value in physics.items():
                            collected[steps]["physics"][name] += value
            for steps in args.refiner_steps:
                key = f"nfe{nfe}_refine{steps}"
                rows = collected[steps]["rows"]
                results[key] = score_summary(
                    rows,
                    collected[steps]["physics"],
                )
                cases = max(1, len(rows))
                update = collected[steps]["updates"]
                coordinates = max(1, int(update["coordinates"]))
                results[key].update(
                    {
                        "latent_update_l1_mean_per_case": (
                            update["l1"] / cases
                        ),
                        "latent_update_l2_mean_per_case": (
                            update["l2"] / cases
                        ),
                        "latent_update_rms_per_coordinate": float(
                            np.sqrt(update["squared"] / coordinates)
                        ),
                    }
                )
                case_rows[key] = rows
            for steps in args.refiner_steps:
                key = f"nfe{nfe}_refine{steps}"
                results[key]["wall_clock_seconds"] = (
                    time.perf_counter() - nfe_started
                )
                results[key]["stage_seconds"] = {
                    "sample": sample_seconds,
                    "refine_all_settings": refine_seconds,
                    "decode_all_settings": decode_seconds,
                    "score_all_settings": score_seconds,
                }
                print(json.dumps({key: results[key]}), flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    baseline_key = (
        f"nfe{args.nfe[0]}_refine{min(args.refiner_steps)}"
    )
    baseline = results[baseline_key]["legalized_hard_cost"]
    best_key = min(
        results,
        key=lambda key: results[key].get("legalized_hard_cost", float("inf")),
    )
    improvement = (baseline - results[best_key]["legalized_hard_cost"]) / baseline if baseline else 0.0
    report = {
        "status": refiner_comparison_status(len(results), improvement),
        "scope": "single-sample inference-only refiner comparison",
        "checkpoint": str(args.checkpoint),
        "ensemble_checkpoint": (
            str(args.ensemble_checkpoint)
            if args.ensemble_checkpoint is not None
            else None
        ),
        "cache": str(args.cache),
        "legalizer_time_limit_ms": args.legalizer_time_limit_ms,
        "legalizer_glop_threads": 1,
        "case_indices": str(args.case_indices) if args.case_indices else None,
        "selected_cases": sum(
            result["cases"] for result in results.values()
        )
        // max(1, len(results)),
        "device": str(device),
        "seed": int(args.seed),
        "legalize_workers": int(args.workers),
        "use_ema": bool(args.use_ema),
        "deterministic": bool(args.deterministic),
        "projector": getattr(model, "projector", None),
        "projector_iterations": getattr(model, "projector_iterations", None),
        "projector_boundary_axes": getattr(model, "projector_boundary_axes", None),
        "projector_boundary_lock_corners": getattr(model, "projector_boundary_lock_corners", None),
        "post_refiner_projector_iterations": args.post_refiner_projector_iterations,
        "post_refiner_projector_strength": args.post_refiner_projector_strength,
        "guidance_scale": args.guidance_scale,
        "guidance_energy": args.guidance_energy,
        "nfe": args.nfe,
        "refiner_steps": args.refiner_steps,
        "refiner": {
            "weights": refiner_weights,
            "learning_rate": args.refiner_lr,
            "schedule": refiner_schedule,
            "schedule_json": (
                str(args.refiner_schedule_json)
                if args.refiner_schedule_json is not None
                else None
            ),
            "shape": args.refiner_shape,
            "optimizer": args.refiner_optimizer,
            "compile_loop": args.refiner_compile_loop,
            "chunk": args.refiner_chunk,
        },
        "best_by_preregistered_mean_metric": best_key,
        "hard_cost_improvement_vs_baseline": improvement,
        "baseline_key": baseline_key,
        "best_of_n_used": False,
        "target_derived_control": False,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    cases_path = args.output.with_name(
        f"{args.output.stem}_cases.json"
    )
    cases_path.write_text(
        json.dumps(case_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "best": best_key,
                "improvement": improvement,
            },
            indent=2,
        )
    )
if __name__ == "__main__":
    main()

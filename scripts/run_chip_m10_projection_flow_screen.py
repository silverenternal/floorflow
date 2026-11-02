#!/usr/bin/env python3
"""Screen projection-consistent per-block flow at a fixed inference budget."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.refiner_config import (
    configure_chip_projector,
)
from floorflow.chip_flow.reproducibility import (
    enable_deterministic_execution,
    stable_batch_seed,
)
from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.evaluation.paired_bootstrap import paired_delta
from scripts.evaluate_chip_refiner import (
    aggregate_physics,
    make_instances,
    score_boxes,
    score_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_m10_projection_consistent_flow_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m10_projection_consistent_flow_v1.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--case-indices", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--stage",
        choices=("quick", "confirmation", "full"),
        default="quick",
    )
    parser.add_argument("--nfe", type=int, nargs="+", default=None)
    parser.add_argument("--refiner-steps", type=int, nargs="+", default=None)
    parser.add_argument("--projection-mode", nargs="+", default=None)
    parser.add_argument("--blocks", type=int, nargs="+", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(relative: str | None) -> Path | None:
    return ROOT / relative if relative else None


def load_case_indices(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    payload = load_json(path)
    values = payload.get("indices", payload) if isinstance(payload, dict) else payload
    return {int(value) for value in values}


def selected_case_indices(
    store: PackedCaseStore,
    requested: set[int] | None,
    blocks: list[int] | None = None,
) -> dict[int, list[int]]:
    selected = {}
    for count in sorted(store.by_block_count):
        if blocks is not None and count not in blocks:
            continue
        indices = [
            int(index)
            for index in store.indices(count)
            if requested is None or int(index) in requested
        ]
        if indices:
            selected[count] = indices
    if requested is not None:
        found = {index for indices in selected.values() for index in indices}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"case manifest indices absent from cache: {missing}")
    return selected


def add_case_fields(
    rows: list[dict],
    batch: dict,
    *,
    mode: str,
    nfe: int,
    refiner_steps: int,
) -> None:
    for position, row in enumerate(rows):
        row.update(
            {
                "index": int(batch["_indices"][position]),
                "mode": mode,
                "nfe": nfe,
                "refiner_steps": refiner_steps,
                "blocks": int(batch["n"]),
            }
        )


def distribution_summary(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        return {"cases": 0}
    costs = np.asarray(
        [float(row["legalized_hard_cost"]) for row in rows],
        dtype=np.float64,
    )
    v_rel = np.asarray(
        [float(row["legalized_v_rel"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "cases": len(rows),
        "cost_p50": float(np.quantile(costs, 0.50)),
        "cost_p90": float(np.quantile(costs, 0.90)),
        "cost_p95": float(np.quantile(costs, 0.95)),
        "v_rel_p50": float(np.quantile(v_rel, 0.50)),
        "v_rel_p90": float(np.quantile(v_rel, 0.90)),
        "v_rel_p95": float(np.quantile(v_rel, 0.95)),
    }


def collect_arm(
    model,
    store: PackedCaseStore,
    selected: dict[int, list[int]],
    config: dict,
    args: argparse.Namespace,
    executor: ProcessPoolExecutor | None,
) -> tuple[dict[str, dict], dict[str, list[dict]], int]:
    device = torch.device(args.device)
    batch_size = int(args.batch_size)
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    refiner = config["refiner"]
    max_refiner = max(int(step) for step in config["refiner_steps"])
    results: dict[str, dict] = {}
    rows_by_key: dict[str, list[dict]] = {}
    cases = 0
    with torch.inference_mode():
        for indices in selected.values():
            for start in range(0, len(indices), batch_size):
                if args.max_batches is not None and cases >= args.max_batches:
                    break
                batch_indices = indices[start : start + batch_size]
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
                for nfe in config["nfes"]:
                    for mode in config["projection_modes"]:
                        generator.manual_seed(
                            stable_batch_seed(int(config["seed"]), batch_indices)
                        )
                        latent, _ = sample_for_batch(
                            model,
                            batch,
                            sampling_steps=int(nfe),
                            generator=generator,
                            projection_mode=mode,
                        )
                        snapshots = {0: latent}
                        if max_refiner > 0:
                            snapshots = refine_closed(
                                latent,
                                refine_case,
                                refiner["weights"],
                                max_refiner,
                                float(refiner["learning_rate"]),
                                copy.deepcopy(refiner["schedule"]),
                                snapshots=tuple(config["refiner_steps"]),
                                chunk=int(refiner["chunk"]),
                                compile_loop=False,
                                shape=bool(refiner["shape"]),
                                optimizer=refiner["optimizer"],
                            )
                        for steps in config["refiner_steps"]:
                            refined = latent if int(steps) == 0 else snapshots[int(steps)]
                            boxes, physics = aggregate_physics(refined, batch)
                            rows = score_boxes(
                                boxes,
                                batch,
                                executor,
                                int(config["legalizer_time_limit_ms"]),
                            )
                            add_case_fields(
                                rows,
                                batch,
                                mode=mode,
                                nfe=int(nfe),
                                refiner_steps=int(steps),
                            )
                            key = f"{mode}_nfe{nfe}_refine{steps}"
                            rows_by_key.setdefault(key, []).extend(rows)
                            entry = results.setdefault(
                                key,
                                {"physics": _empty_physics()},
                            )
                            for name, value in physics.items():
                                entry["physics"][name] += value
                cases += len(batch_indices)
    summaries = {}
    for key, rows in rows_by_key.items():
        summaries[key] = score_summary(rows, results[key]["physics"])
        summaries[key]["distribution"] = distribution_summary(rows)
        summaries[key]["block_strata"] = {
            "21_40": distribution_summary(
                [row for row in rows if int(row["blocks"]) <= 40]
            ),
            "41_80": distribution_summary(
                [
                    row
                    for row in rows
                    if 41 <= int(row["blocks"]) <= 80
                ]
            ),
            "81_120": distribution_summary(
                [row for row in rows if int(row["blocks"]) > 80]
            ),
        }
    return summaries, rows_by_key, cases


def compare_modes(
    summaries: dict[str, dict],
    rows_by_key: dict[str, list[dict]],
    baseline_mode: str,
    config: dict,
) -> dict[str, dict]:
    comparisons = {}
    for nfe in config["nfes"]:
        for steps in config["refiner_steps"]:
            baseline_key = f"{baseline_mode}_nfe{nfe}_refine{steps}"
            baseline_rows = rows_by_key[baseline_key]
            for mode in config["projection_modes"]:
                if mode == baseline_mode:
                    continue
                current_key = f"{mode}_nfe{nfe}_refine{steps}"
                paired = {}
                for metric in (
                    "legalized_hard_cost",
                    "legalized_v_rel",
                    "legalized_area_gap",
                    "legalized_grouping",
                    "legalized_boundary",
                ):
                    paired[metric] = paired_delta(
                        rows_by_key[current_key],
                        baseline_rows,
                        metric,
                        iterations=10000,
                        seed=20261031,
                        lower_is_better=True,
                    )
                mean_delta = summaries[current_key][
                    "legalized_hard_cost"
                ] - summaries[baseline_key]["legalized_hard_cost"]
                comparisons[current_key] = {
                    "baseline_mode": baseline_mode,
                    "mean_cost_delta": mean_delta,
                    "paired": paired,
                    "mean_pass": (
                        mean_delta < 0.0
                        and summaries[current_key]["legalized_v_rel"]
                        < summaries[baseline_key]["legalized_v_rel"]
                        and summaries[current_key][
                            "legalized_feasible_fraction"
                        ]
                        == 1.0
                    ),
                }
    return comparisons


def _empty_physics() -> dict[str, float]:
    return {
        "overlap": 0.0,
        "mib": 0.0,
        "boundary": 0.0,
        "wl": 0.0,
        "area": 0.0,
        "total": 0.0,
        "group": 0.0,
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if args.stage != "full":
        config.update(config["stages"][args.stage])
    if args.nfe is not None:
        config["nfes"] = args.nfe
    if args.refiner_steps is not None:
        config["refiner_steps"] = args.refiner_steps
    if args.projection_mode is not None:
        config["projection_modes"] = args.projection_mode
    if config["primary_baseline_mode"] not in config["projection_modes"]:
        raise ValueError("projection modes must include the primary baseline mode")
    checkpoint_path = args.checkpoint or resolve(config["checkpoint"])
    cache_path = args.cache or resolve(config["cache"])
    case_path = args.case_indices or resolve(config["case_indices"])
    args.device = args.device or config["device"]
    args.batch_size = args.batch_size or int(config["batch_size"])
    args.workers = args.workers or int(config["legalizer_workers"])
    if bool(config["deterministic"]):
        enable_deterministic_execution()
    store = PackedCaseStore(cache_path)
    selected = selected_case_indices(
        store,
        load_case_indices(case_path),
        args.blocks,
    )
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model = load_chip_checkpoint(
        checkpoint,
        args.device,
        use_ema=bool(config["use_ema"]),
    )
    if not isinstance(model, HierarchicalTokenFlowModel):
        raise TypeError("M10 requires the typed hierarchical flow model")
    model = configure_chip_projector(
        model,
        enabled=True,
        iterations=int(config["projector_iterations"]),
        boundary_axes=None,
        boundary_lock_corners=None,
    )
    executor = (
        ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            mp_context=multiprocessing.get_context("spawn"),
        )
        if args.workers > 1
        else None
    )
    started = time.perf_counter()
    try:
        summaries, rows, cases = collect_arm(
            model,
            store,
            selected,
            config,
            args,
            executor,
        )
    finally:
        if executor is not None:
            executor.shutdown()
    comparisons = compare_modes(
        summaries,
        rows,
        config["primary_baseline_mode"],
        config,
    )
    report = {
        "version": "chip-m10-projection-consistent-flow-report-v1",
        "status": "complete",
        "stage": args.stage,
        "config": str(args.config),
        "checkpoint": str(checkpoint_path),
        "cache": str(cache_path),
        "case_indices": str(case_path) if case_path else None,
        "device": args.device,
        "cases": cases,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "nfes": config["nfes"],
        "refiner_steps": config["refiner_steps"],
        "projection_modes": config["projection_modes"],
        "max_refiner_steps_per_trajectory": max(config["refiner_steps"]),
        "elapsed_seconds": time.perf_counter() - started,
        "summaries": summaries,
        "comparisons": comparisons,
        "any_mean_pass": any(
            comparison["mean_pass"] for comparison in comparisons.values()
        ),
        "claim_boundary": (
            "Frozen-subset mechanism screen only. It cannot support a SOTA or "
            "routed-PPA claim without the matched full-data and external gates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, report)
    rows_path = args.output.with_name(f"{args.output.stem}_cases.json")
    write_json_atomic(rows_path, {"rows": rows})
    print(json.dumps({key: report[key] for key in report if key != "summaries"}))


if __name__ == "__main__":
    main()

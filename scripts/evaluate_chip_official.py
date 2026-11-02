#!/usr/bin/env python3
"""Evaluate one chip-flow checkpoint with Trinity's full FloorSet scorer."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party/Trinity/src"))

from trinity.floorplan.legalize.scale_pack import last_info as legalizer_last_info
from trinity.floorplan.scoring.cost import score_full_fast
from trinity.floorplan.types import FloorplanInstance, Placement

from floorflow.chip_flow.batched import batch_physics_terms
from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.hierarchical_model import ChipHierarchicalSlotFlowModel
from floorflow.chip_flow.legalization import scale_pack_legalize_reproducible
from floorflow.chip_flow.legalizer_policy import LegalizerPolicy
from floorflow.chip_flow.model import ChipBlockFlowModel, decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.proposals import token_ot_reorder
from floorflow.chip_flow.reproducibility import enable_deterministic_execution
from floorflow.chip_flow.score_diagnostics import score_diagnostics
from floorflow.chip_flow.slot_model import ChipSlotFlowModel
from floorflow.chip_flow.sparse_flow_model import ChipSparseHierarchyFlowModel
from floorflow.chip_flow.sparse_slot_flow import ChipSlotTransportFlowModel
from floorflow.chip_flow.trinity_token_flow import ChipTrinityTokenFlowModel
from floorflow.chip_physics import group_penalty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--cases-per-count", type=int, default=0)
    parser.add_argument("--blocks", type=int, nargs="+", default=None)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--legalize", action="store_true")
    parser.add_argument("--save-cases", action="store_true")
    parser.add_argument(
        "--save-raw-boxes",
        action="store_true",
        help="Include predicted xywh in per-case rows for legalizer counterfactuals.",
    )
    parser.add_argument(
        "--save-legalized-boxes",
        action="store_true",
        help="Include legalized xywh in per-case rows for candidate distillation.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--legalizer-time-limit-ms",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--legalizer-reshape",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--legalizer-trust", type=float, default=1.3)
    parser.add_argument(
        "--legalizer-relation",
        choices=("restored", "expanded"),
        default="restored",
    )
    parser.add_argument(
        "--legalizer-outline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def select_indices(
    store: PackedCaseStore,
    per_count: int,
    blocks: list[int] | None = None,
) -> np.ndarray:
    if per_count <= 0:
        return store.indices()
    selected = []
    for count in sorted(store.by_block_count):
        if blocks is not None and count not in blocks:
            continue
        candidates = store.indices(count)
        selected.extend(candidates[: min(per_count, len(candidates))].tolist())
    return np.asarray(selected, dtype=np.int64)


def score_case(payload: dict) -> dict:
    """Score and optionally legalize one case in an independent CPU process."""

    count = int(payload["blocks"])
    boxes = payload["boxes"]
    gt_boxes = payload["gt_boxes"]
    constraints = payload["constraints"]
    targets = np.full((count, 4), -1.0, dtype=np.float32)
    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    targets[fixed, 2:] = gt_boxes[fixed, 2:]
    targets[preplaced] = gt_boxes[preplaced]
    instance = FloorplanInstance(
        block_count=count,
        area_targets=payload["areas"],
        constraints=constraints,
        b2b=payload["b2b"],
        p2b=payload["p2b"],
        pins_pos=payload["pins"],
        target_positions=targets,
        gt_positions=payload["gt_boxes"],
    )
    placement = Placement(xywh=boxes, instance=instance)
    score = score_full_fast(placement)
    row = {
        "index": int(payload["index"]),
        "blocks": count,
        "raw_hard_cost": score.cost,
        "raw_hpwl_gap": score.hpwl_gap,
        "raw_area_gap": score.area_gap,
        "raw_v_rel": score.v_rel,
        "raw_feasible": bool(score.feasible),
        "latent_mse": float(payload["latent_mse"]),
    }
    row.update(score_diagnostics(
        "raw", score, boxes, gt_boxes,
        payload["b2b"], payload["p2b"], payload["pins"],
    ))
    if payload.get("return_raw_boxes", False):
        row["raw_xywh"] = np.asarray(boxes, dtype=np.float32).tolist()
        for key in ("gt_boxes", "areas", "constraints", "b2b", "p2b", "pins"):
            row[key] = np.asarray(payload[key], dtype=np.float32).tolist()
    if payload["legalize"]:
        legalizer_options = dict(payload.get("legalizer_options", {}))
        legalized = scale_pack_legalize_reproducible(
            placement,
            time_limit_ms=int(payload.get("legalizer_time_limit_ms", 2000)),
            **legalizer_options,
        )
        legalized_score = score_full_fast(legalized)
        row.update(
            {
                "legalized_hard_cost": legalized_score.cost,
                "legalized_hpwl_gap": legalized_score.hpwl_gap,
                "legalized_area_gap": legalized_score.area_gap,
                "legalized_v_rel": legalized_score.v_rel,
                "legalized_feasible": bool(legalized_score.feasible),
                "legalizer_rung": int(legalizer_last_info()["rung"]),
                "legalizer_solves": int(legalizer_last_info()["solves"]),
                "legalizer_ms": float(legalizer_last_info()["ms"]),
                "legalizer_reshaped": int(legalizer_last_info()["reshaped"]),
            }
        )
        row.update(score_diagnostics(
            "legalized", legalized_score, np.asarray(legalized.xywh), gt_boxes,
            payload["b2b"], payload["p2b"], payload["pins"],
        ))
        if payload.get("return_legalized_boxes", False):
            row["legalized_xywh"] = np.asarray(
                legalized.xywh,
                dtype=np.float32,
            ).tolist()
    return row


def case_payload(
    batch: dict,
    index: int,
    predicted_boxes: torch.Tensor,
    latent_mse: float,
    *,
    legalize: bool,
    legalizer_time_limit_ms: int | None = None,
    legalizer_options: dict | None = None,
    return_raw_boxes: bool = False,
    return_legalized_boxes: bool = False,
) -> dict:
    count = int(batch["n"])
    return {
        "index": int(batch["_indices"][index]),
        "blocks": count,
        "boxes": predicted_boxes[:count].cpu().numpy().copy(),
        "gt_boxes": batch["boxes"][index, :count].cpu().numpy().copy(),
        "areas": batch["areas"][index, :count].cpu().numpy().copy(),
        "constraints": batch["constraints"][index, :count].cpu().numpy().copy(),
        "b2b": batch["b2b"][index, : int(batch["n_b2b"][index])]
        .cpu()
        .numpy()
        .copy(),
        "p2b": batch["p2b"][index, : int(batch["n_p2b"][index])]
        .cpu()
        .numpy()
        .copy(),
        "pins": batch["pins"][index, : int(batch["n_pins"][index])]
        .cpu()
        .numpy()
        .copy(),
        "latent_mse": latent_mse,
        "legalize": legalize,
        "legalizer_time_limit_ms": (
            2000
            if legalizer_time_limit_ms is None
            else int(legalizer_time_limit_ms)
        ),
        "legalizer_options": dict(legalizer_options or {}),
        "return_raw_boxes": bool(return_raw_boxes),
        "return_legalized_boxes": bool(return_legalized_boxes),
    }


@torch.no_grad()
def evaluate(
    store: PackedCaseStore,
    indices: np.ndarray,
    model: (
        ChipBlockFlowModel
        | ChipSlotFlowModel
        | ChipSparseHierarchyFlowModel
        | ChipSlotTransportFlowModel
        | ChipTrinityTokenFlowModel
        | HierarchicalTokenFlowModel
    ),
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    totals: dict[str, float] = {}
    cases = 0
    feasible = {"raw": 0, "legalized": 0}
    per_block = {"raw": {}, "legalized": {}}
    active_slot_counts: list[int] = []
    router_budget_means: list[float] = []
    router_budget_mins: list[float] = []
    router_budget_maxes: list[float] = []
    router_top_ks: list[float] = []
    router_load_maxes: list[float] = []
    routing_entropies: list[float] = []
    case_rows = []
    generator = torch.Generator(device=args.device).manual_seed(19)
    legalizer_policy = LegalizerPolicy(
        relation=args.legalizer_relation,
        reshape=args.legalizer_reshape,
        trust=args.legalizer_trust,
        outline=args.legalizer_outline,
        time_limit_ms=args.legalizer_time_limit_ms,
    )
    legalizer_policy.validate()
    executor = (
        ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            mp_context=multiprocessing.get_context("spawn"),
        )
        if args.workers > 1
        else None
    )
    try:
        for count in sorted(set(store.arrays["n"][indices].tolist())):
            count_indices = indices[store.arrays["n"][indices] == count]
            for start in range(0, len(count_indices), args.batch_size):
                batch_indices = count_indices[start : start + args.batch_size]
                batch = store.batch(
                    batch_indices,
                    device=args.device,
                )
                batch["_indices"] = torch.as_tensor(
                    batch_indices,
                    device=args.device,
                )
                if isinstance(model, ChipHierarchicalSlotFlowModel):
                    active_slot_counts.extend(
                        model.active_slot_counts(batch["features"]).cpu().tolist()
                    )
                if isinstance(
                    model,
                    (
                        ChipSlotFlowModel,
                        ChipSparseHierarchyFlowModel,
                        ChipSlotTransportFlowModel,
                        ChipTrinityTokenFlowModel,
                        HierarchicalTokenFlowModel,
                    ),
                ):
                    latent, diagnostics = sample_for_batch(
                        model,
                        batch,
                        sampling_steps=args.sampling_steps,
                        generator=generator,
                    )
                    if "active_slot_count" in diagnostics:
                        active_slot_counts.extend(
                            diagnostics["active_slot_count"].cpu().tolist()
                        )
                        router_budget_means.extend(
                            diagnostics["router_budget_mean"]
                            .expand_as(
                                diagnostics["active_slot_count"]
                            )
                            .float()
                            .cpu()
                            .tolist()
                        )
                        router_budget_mins.extend(
                            diagnostics["router_budget_min"]
                            .expand_as(
                                diagnostics["active_slot_count"]
                            )
                            .float()
                            .cpu()
                            .tolist()
                        )
                        router_budget_maxes.extend(
                            diagnostics["router_budget_max"]
                            .expand_as(
                                diagnostics["active_slot_count"]
                            )
                            .float()
                            .cpu()
                            .tolist()
                        )
                        top_k = diagnostics.get(
                            "router_budget_per_case",
                            diagnostics["router_budget_mean"].expand(
                                diagnostics["active_slot_count"].shape[0]
                            ),
                        )
                        router_top_ks.extend(top_k.float().cpu().tolist())
                        router_load = diagnostics["router_load"].float()
                        router_load_maxes.extend(
                            router_load.amax(dim=1).cpu().tolist()
                        )
                        routing_entropies.extend(
                            diagnostics["router_usage_entropy"]
                            .expand(
                                diagnostics["active_slot_count"].shape[0]
                            )
                            .float()
                            .cpu()
                            .tolist()
                        )
                    latent = project_conditioned_latent(
                        latent,
                        batch["target_latent"],
                        batch["constraints"],
                    )
                else:
                    latent = project_conditioned_latent(
                        batch["source_latent"],
                        batch["target_latent"],
                        batch["constraints"],
                    )
                    for step in range(args.sampling_steps):
                        time = torch.full(
                            (latent.shape[0],),
                            step / args.sampling_steps,
                            device=args.device,
                        )
                        latent = latent + model(
                            latent,
                            time,
                            batch["features"],
                        ) / args.sampling_steps
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
                aligned = torch.stack(
                    [
                        token_ot_reorder(
                            latent[index].detach().cpu(),
                            batch["target_latent"][index].detach().cpu(),
                        )
                        for index in range(latent.shape[0])
                    ]
                ).to(args.device)
                target_aspect = batch["target_latent"][..., 2]
                values = {
                    "latent_mse": (latent - aligned).pow(2).mean(dim=(1, 2)),
                    "rho_mae": (latent[..., 2] - aligned[..., 2])
                    .abs()
                    .mean(dim=1),
                    "aspect_diversity_ratio": (
                        latent[..., 2].std(dim=1, correction=0)
                        / target_aspect.std(dim=1, correction=0).clamp_min(
                            1.0e-6
                        )
                    ),
                    "overlap": terms["overlap"],
                    "group": group,
                    "mib": terms["mib"],
                    "boundary": terms["boundary"],
                    "wl": terms["wl"],
                    "area": terms["area"],
                }
                payloads = [
                    case_payload(
                        batch,
                        index,
                        boxes[index],
                        float(values["latent_mse"][index]),
                        legalize=args.legalize,
                        legalizer_time_limit_ms=args.legalizer_time_limit_ms,
                        legalizer_options=legalizer_policy.kwargs(),
                        return_raw_boxes=args.save_raw_boxes,
                        return_legalized_boxes=args.save_legalized_boxes,
                    )
                    for index in range(latent.shape[0])
                ]
                rows = list(
                    executor.map(score_case, payloads)
                    if executor is not None
                    else map(score_case, payloads)
                )
                for row in rows:
                    row_count = int(row["blocks"])
                    feasible["raw"] += int(row["raw_feasible"])
                    per_block["raw"].setdefault(row_count, []).append(
                        float(row["raw_hard_cost"])
                    )
                    if args.legalize:
                        feasible["legalized"] += int(row["legalized_feasible"])
                        per_block["legalized"].setdefault(row_count, []).append(
                            float(row["legalized_hard_cost"])
                        )
                    for name in (
                        "raw_hard_cost",
                        "raw_hpwl_gap",
                        "raw_area_gap",
                        "raw_v_rel",
                    ):
                        totals[name] = totals.get(name, 0.0) + float(row[name])
                    if args.legalize:
                        for name in (
                            "legalized_hard_cost",
                            "legalized_hpwl_gap",
                            "legalized_area_gap",
                            "legalized_v_rel",
                        ):
                            totals[name] = totals.get(name, 0.0) + float(row[name])
                    case_rows.append(row)
                for name, value in values.items():
                    totals[name] = totals.get(name, 0.0) + float(value.sum())
                cases += latent.shape[0]
    finally:
        if executor is not None:
            executor.shutdown()
    summary = {name: value / cases for name, value in totals.items()}
    summary["raw_feasible_fraction"] = feasible["raw"] / cases
    if args.legalize:
        summary["legalized_feasible_fraction"] = feasible["legalized"] / cases
    summary["cases"] = cases
    if active_slot_counts:
        summary["active_slot_count_mean"] = float(np.mean(active_slot_counts))
        summary["active_slot_count_min"] = int(np.min(active_slot_counts))
        summary["active_slot_count_max"] = int(np.max(active_slot_counts))
    if router_budget_means:
        summary["router_budget_mean"] = float(np.mean(router_budget_means))
        summary["router_budget_min"] = float(np.min(router_budget_mins))
        summary["router_budget_max"] = float(np.max(router_budget_maxes))
    if router_top_ks:
        summary["top_k_mean"] = float(np.mean(router_top_ks))
        summary["top_k_min"] = float(np.min(router_top_ks))
        summary["top_k_max"] = float(np.max(router_top_ks))
    if router_load_maxes:
        summary["router_load_max"] = float(np.max(router_load_maxes))
        summary["router_load_mean"] = float(np.mean(router_load_maxes))
        summary["routing_entropy_mean"] = float(np.mean(routing_entropies))
    summary["hard_cost_by_block"] = {
        stage: {
            str(count): float(np.mean(values))
            for count, values in stage_values.items()
        }
        for stage, stage_values in per_block.items()
    }
    return summary, case_rows


def main() -> None:
    args = parse_args()
    if args.deterministic:
        enable_deterministic_execution()
    started = time.perf_counter()
    store = PackedCaseStore(args.cache)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = load_chip_checkpoint(
        checkpoint,
        args.device,
        use_ema=args.use_ema,
    )
    result, case_rows = evaluate(
        store,
        select_indices(store, args.cases_per_count, args.blocks),
        model,
        args,
    )
    report = {
        "status": "complete", "evaluation_seconds": time.perf_counter() - started,
        "metric_schema": "plan_c_official_v3",
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "config": json.loads(json.dumps(checkpoint["config"], default=str)),
        "evaluation": json.loads(json.dumps(vars(args), default=str)),
        "use_ema": args.use_ema,
        "deterministic": args.deterministic,
        "sampling_steps": args.sampling_steps,
        "inference_seed": 19,
        "legalizer_glop_threads": 1,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.save_cases:
        cases_path = args.output.with_name(f"{args.output.stem}_cases.json")
        cases_path.write_text(
            json.dumps(case_rows, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({key: value for key, value in result.items() if key != "hard_cost_by_block"}, indent=2))


if __name__ == "__main__":
    main()

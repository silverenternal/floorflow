#!/usr/bin/env python3
"""Attribute chip endpoint degradation to refiner and legalizer displacement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.legalize.scale_pack import last_info
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed

from floorflow.chip_flow.boundary_snap import snap_boundary_boxes
from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.hard_constraint_projector import (
    _boundary_violation_count,
    _overlap_pairs,
)
from floorflow.chip_flow.refiner_config import (
    configure_chip_projector,
    parse_refiner_weight_overrides,
)
from floorflow.chip_flow.reproducibility import (
    enable_deterministic_execution,
    stable_batch_seed,
)
from scripts.evaluate_chip_refiner import (
    aggregate_physics,
    case_payload,
    make_instances,
    score_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/chip_hierarchy/m27_training_seed_replication_v1"
            / "m27_seed21_5k/checkpoints/final.pt"
        ),
    )
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
    parser.add_argument("--refiner-steps", type=int, default=400)
    parser.add_argument("--refiner-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--refiner-schedule",
        choices=("linear", "cosine", "constant"),
        default="linear",
    )
    parser.add_argument("--refiner-optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--refiner-chunk", type=int, default=5)
    parser.add_argument("--refiner-weight", action="append", default=[])
    parser.add_argument("--projector-iterations", type=int, default=8)
    parser.add_argument("--legalizer-time-limit-ms", type=int, default=2000)
    parser.add_argument("--legalizer-mu", type=float, default=0.5)
    parser.add_argument("--legalizer-penalty", type=float, default=10.0)
    parser.add_argument(
        "--legalizer-relation",
        choices=("restored", "expanded"),
        default="restored",
    )
    parser.add_argument("--snap-boundary", action="store_true")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_indices(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("indices", payload) if isinstance(payload, dict) else payload
    indices = [int(value) for value in values]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("case manifest must contain unique non-empty indices")
    return indices


def displacement(
    left: torch.Tensor,
    right: torch.Tensor,
    constraints: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float]:
    left = left.float()
    right = right.float()
    left_center = left[..., :2] + left[..., 2:] * 0.5
    right_center = right[..., :2] + right[..., 2:] * 0.5
    delta = right_center - left_center
    distance = torch.linalg.vector_norm(delta, dim=-1) / scale[:, None]
    codes = constraints[..., 4].long()
    boundary = codes.ne(0)
    anchored = constraints[..., :2].ne(0).any(dim=-1)
    shape_change = (right[..., 2:] - left[..., 2:]).abs().sum(dim=-1)
    return {
        "center_l1": float((delta.abs().sum(dim=-1) / scale[:, None]).sum()),
        "center_l2": float(distance.sum()),
        "center_l2_max": float(distance.max()),
        "center_l2_mean": float(distance.mean()),
        "boundary_center_l2": float(distance[boundary].sum()),
        "boundary_blocks": int(boundary.sum()),
        "anchored_center_l2": float(distance[anchored].sum()),
        "anchored_blocks": int(anchored.sum()),
        "shape_l1": float(shape_change.sum()),
    }


def stage_metrics(
    boxes: torch.Tensor,
    batch: dict,
    index: int,
) -> dict[str, float]:
    row = score_case(
        case_payload(
            batch,
            index,
            boxes,
            0.0,
            legalize=False,
        )
    )
    return {
        "cost": float(row["raw_hard_cost"]),
        "hpwl_gap": float(row["raw_hpwl_gap"]),
        "area_gap": float(row["raw_area_gap"]),
        "v_rel": float(row["raw_v_rel"]),
        "b2b_hpwl_gap": float(row["raw_b2b_hpwl_gap"]),
        "p2b_hpwl_gap": float(row["raw_p2b_hpwl_gap"]),
        "rho_mae": float(row["raw_rho_mae"]),
        "grouping": float(row["raw_grouping"]),
        "boundary": float(row["raw_boundary"]),
        "feasible": bool(row["raw_feasible"]),
    }


def summarize(rows: list[dict]) -> dict:
    def mean(*path: str) -> float:
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values))

    raw_to_refined = np.asarray(
        [row["displacement"]["raw_to_refined"]["center_l2"] for row in rows]
    )
    refined_to_legal = np.asarray(
        [row["displacement"]["refined_to_legalized"]["center_l2"] for row in rows]
    )
    total = raw_to_refined + refined_to_legal
    legalizer_fraction = refined_to_legal / np.maximum(total, 1.0e-9)
    cost_change = np.asarray(
        [
            row["stages"]["legalized"]["cost"]
            - row["stages"]["refined"]["cost"]
            for row in rows
        ]
    )
    rho = (
        spearmanr(refined_to_legal, cost_change)
        if len(rows) > 1
        else None
    )
    bins = {}
    for label, low, high in (
        ("21-40", 21, 40),
        ("41-80", 41, 80),
        ("81-100", 81, 100),
        ("101-120", 101, 120),
    ):
        selected = [row for row in rows if low <= row["blocks"] <= high]
        if selected:
            bins[label] = {
                "cases": len(selected),
                "refined_cost": float(
                    np.mean([row["stages"]["refined"]["cost"] for row in selected])
                ),
                "legalized_cost": float(
                    np.mean([row["stages"]["legalized"]["cost"] for row in selected])
                ),
                "refined_v_rel": float(
                    np.mean([row["stages"]["refined"]["v_rel"] for row in selected])
                ),
                "legalized_v_rel": float(
                    np.mean([row["stages"]["legalized"]["v_rel"] for row in selected])
                ),
            }
    worst = sorted(
        rows,
        key=lambda row: (
            row["stages"]["legalized"]["cost"]
            - row["stages"]["refined"]["cost"]
        ),
        reverse=True,
    )[:10]
    return {
        "cases": len(rows),
        "raw_cost": mean("stages", "raw", "cost"),
        "refined_cost": mean("stages", "refined", "cost"),
        "legalized_cost": mean("stages", "legalized", "cost"),
        "raw_v_rel": mean("stages", "raw", "v_rel"),
        "refined_v_rel": mean("stages", "refined", "v_rel"),
        "legalized_v_rel": mean("stages", "legalized", "v_rel"),
        "raw_overlap_pairs": mean("overlap", "raw"),
        "refined_overlap_pairs": mean("overlap", "refined"),
        "raw_boundary_violations": mean("boundary", "raw"),
        "refined_boundary_violations": mean("boundary", "refined"),
        "legalized_boundary_violations": mean("boundary", "legalized"),
        "raw_to_refined_center_l2": float(raw_to_refined.mean()),
        "refined_to_legalized_center_l2": float(refined_to_legal.mean()),
        "legalizer_displacement_fraction": float(legalizer_fraction.mean()),
        "refiner_legalizer_spearman": (
            float(rho.statistic) if rho is not None else None
        ),
        "refiner_legalizer_spearman_p": (
            float(rho.pvalue) if rho is not None else None
        ),
        "block_bins": bins,
        "worst_legalizer_regressions": [
            {
                "index": row["index"],
                "blocks": row["blocks"],
                "refined_cost": row["stages"]["refined"]["cost"],
                "legalized_cost": row["stages"]["legalized"]["cost"],
                "legalizer_center_l2": row["displacement"][
                    "refined_to_legalized"
                ]["center_l2"],
                "legalizer_rung": row["legalizer"]["rung"],
                "legalizer_solves": row["legalizer"]["solves"],
            }
            for row in worst
        ],
    }


def main() -> None:
    args = parse_args()
    if args.deterministic:
        enable_deterministic_execution()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = load_chip_checkpoint(checkpoint, device, use_ema=True)
    model = configure_chip_projector(
        model,
        enabled=True,
        iterations=args.projector_iterations,
        boundary_axes=False,
        boundary_lock_corners=True,
    )
    store = PackedCaseStore(args.cache)
    indices = load_indices(args.case_indices)
    weights = parse_refiner_weight_overrides(args.refiner_weight)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows = []
    with torch.no_grad():
        for count in sorted({int(store.arrays["n"][index]) for index in indices}):
            group = [index for index in indices if int(store.arrays["n"][index]) == count]
            for start in range(0, len(group), args.batch_size):
                batch_rows = group[start : start + args.batch_size]
                batch = store.batch(np.asarray(batch_rows), device=device)
                batch["_indices"] = torch.as_tensor(batch_rows, device=device)
                generator.manual_seed(stable_batch_seed(args.seed, batch_rows))
                raw_latent, _ = sample_for_batch(
                    model,
                    batch,
                    sampling_steps=args.nfe,
                    generator=generator,
                )
                raw_boxes, _ = aggregate_physics(raw_latent, batch)
                refine_case = build_refine_case(make_instances(batch), 1, device)
                refined_latent = refine_closed(
                    raw_latent,
                    refine_case,
                    weights,
                    args.refiner_steps,
                    args.refiner_lr,
                    args.refiner_schedule,
                    chunk=args.refiner_chunk,
                    compile_loop=False,
                    shape=True,
                    optimizer=args.refiner_optimizer,
                )[args.refiner_steps]
                refined_boxes, _ = aggregate_physics(refined_latent, batch)
                for row, case_index in enumerate(batch_rows):
                    constraints = batch["constraints"][row, :count]
                    raw = raw_boxes[row, :count]
                    refined = refined_boxes[row, :count]
                    payload = case_payload(
                        batch,
                        row,
                        refined,
                        0.0,
                        legalize=True,
                        legalizer_time_limit_ms=args.legalizer_time_limit_ms,
                    )
                    payload["legalizer_options"] = {
                        "mu": args.legalizer_mu,
                        "penalty": args.legalizer_penalty,
                        "relation": args.legalizer_relation,
                    }
                    payload["return_legalized_boxes"] = True
                    official = score_case(payload)
                    info = dict(last_info())
                    legalized_boxes = torch.as_tensor(
                        official["legalized_xywh"],
                        device=device,
                        dtype=torch.float32,
                    )
                    snap = None
                    if args.snap_boundary:
                        snapped = snap_boundary_boxes(
                            legalized_boxes,
                            constraints[:, 4],
                            ~constraints[:, :2].ne(0).any(dim=-1),
                        )
                        legalized_boxes = snapped.boxes
                        snap = {
                            "moves": snapped.moves,
                            "before": snapped.before,
                            "after": snapped.after,
                        }
                    official_row = {
                        "cost": float(official["legalized_hard_cost"]),
                        "hpwl_gap": float(official["legalized_hpwl_gap"]),
                        "area_gap": float(official["legalized_area_gap"]),
                        "v_rel": float(official["legalized_v_rel"]),
                        "b2b_hpwl_gap": float(official["legalized_b2b_hpwl_gap"]),
                        "p2b_hpwl_gap": float(official["legalized_p2b_hpwl_gap"]),
                        "rho_mae": float(official["legalized_rho_mae"]),
                        "grouping": float(official["legalized_grouping"]),
                        "boundary": float(official["legalized_boundary"]),
                        "feasible": bool(official["legalized_feasible"]),
                    }
                    scale = (
                        batch["areas"][row, :count].sum().sqrt().clamp_min(1.0e-6)
                    )
                    legalized_row = (
                        stage_metrics(legalized_boxes, batch, row)
                        if snap is not None and snap["moves"] > 0
                        else official_row
                    )
                    rows.append(
                        {
                            "index": int(case_index),
                            "blocks": count,
                            "stages": {
                                "raw": stage_metrics(raw, batch, row),
                                "refined": stage_metrics(refined, batch, row),
                                "legalized": legalized_row,
                            },
                            "overlap": {
                                "raw": _overlap_pairs(raw[None]),
                                "refined": _overlap_pairs(refined[None]),
                            },
                            "boundary": {
                                "raw": _boundary_violation_count(
                                    raw[None], constraints[None]
                                ),
                                "refined": _boundary_violation_count(
                                    refined[None], constraints[None]
                                ),
                                "legalized": int(
                                    legalized_row["boundary"]
                                ),
                            },
                            "displacement": {
                                "raw_to_refined": displacement(
                                    raw[None],
                                    refined[None],
                                    constraints[None],
                                    scale[None],
                                ),
                                "refined_to_legalized": displacement(
                                    refined[None],
                                    legalized_boxes[None],
                                    constraints[None],
                                    scale[None],
                                ),
                                "raw_to_legalized": displacement(
                                    raw[None],
                                    legalized_boxes[None],
                                    constraints[None],
                                    scale[None],
                                ),
                            },
                            "legalizer": {
                                "rung": int(info["rung"]),
                                "solves": int(info["solves"]),
                                "ms": float(info["ms"]),
                                "reshaped": int(info["reshaped"]),
                            },
                            "boundary_snap": snap,
                        }
                    )
    report = {
        "version": "chip-m48-displacement-attribution-v1",
        "status": "diagnostic_complete",
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "case_indices": str(args.case_indices),
        "protocol": {
            "nfe": args.nfe,
            "refiner_steps": args.refiner_steps,
            "refiner_weights": weights,
            "refiner_lr": args.refiner_lr,
            "refiner_schedule": args.refiner_schedule,
            "projector_iterations": args.projector_iterations,
            "legalizer_time_limit_ms": args.legalizer_time_limit_ms,
            "legalizer_mu": args.legalizer_mu,
            "legalizer_penalty": args.legalizer_penalty,
            "legalizer_relation": args.legalizer_relation,
            "snap_boundary": bool(args.snap_boundary),
            "seed": args.seed,
        },
        "summary": summarize(rows),
        "rows": rows,
        "boundary": "Stage-attribution diagnostic only; no candidate or SOTA claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

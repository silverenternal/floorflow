#!/usr/bin/env python3
"""Evaluate a two-trajectory, target-free chip legalizer portfolio."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.legalize.scale_pack import last_info
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.legal_parent_reflow import (
    critical_block_mask,
    partial_renoise_parent,
)
from floorflow.chip_flow.legalizer_portfolio import (
    LegalizerCandidate,
    select_legalizer_candidate,
)
from floorflow.chip_flow.model import encode_latents
from floorflow.chip_flow.portfolio_router import router_features
from floorflow.chip_flow.refiner_config import (
    configure_chip_projector,
    parse_refiner_weight_overrides,
)
from floorflow.chip_flow.reproducibility import (
    enable_deterministic_execution,
    stable_batch_seed,
)
from floorflow.chip_flow.target_free_geometry import (
    target_free_geometry_metrics,
)
from floorflow.evaluation.paired_bootstrap import paired_delta
from scripts.analyze_chip_m48_displacement import (
    displacement,
    load_indices,
    stage_metrics,
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
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument("--case-indices", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--blocks", type=int, nargs="+", default=None)
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument("--refiner-steps", type=int, default=1600)
    parser.add_argument("--refiner-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--refiner-schedule",
        choices=("linear", "cosine", "constant"),
        default="linear",
    )
    parser.add_argument("--refiner-optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--refiner-chunk", type=int, default=5)
    parser.add_argument("--refiner-weight", action="append", default=[])
    parser.add_argument("--legalizer-time-limit-ms", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--selection-policy",
        choices=("legality_first", "balanced"),
        default="legality_first",
    )
    parser.add_argument("--legal-parent-reflow", action="store_true")
    parser.add_argument("--single-parent-reflow", action="store_true")
    parser.add_argument("--reflow-top-fraction", type=float, default=0.2)
    parser.add_argument("--reflow-noise-fraction", type=float, default=0.25)
    parser.add_argument("--reflow-steps", type=int, default=2)
    parser.add_argument(
        "--reflow-selection-policy",
        choices=(
            "legality_first",
            "balanced",
            "geometry_consensus",
            "legality_guarded_consensus",
        ),
        default="legality_guarded_consensus",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def legalized_candidate(
    name: str,
    refined: torch.Tensor,
    refined_boxes: torch.Tensor,
    batch: dict,
    row: int,
    constraints: torch.Tensor,
    scale: torch.Tensor,
    *,
    relation: str,
    time_limit_ms: int,
    device: torch.device,
) -> dict:
    count = int(batch["n"])
    refined_boxes_row = refined_boxes[row, :count]
    payload = case_payload(
        batch,
        row,
        refined_boxes_row,
        0.0,
        legalize=True,
        legalizer_time_limit_ms=time_limit_ms,
    )
    payload["legalizer_options"] = {
        "mu": 0.5,
        "penalty": 10.0,
        "relation": relation,
    }
    payload["return_legalized_boxes"] = True
    official = score_case(payload)
    legalized_boxes = torch.as_tensor(
        official["legalized_xywh"],
        device=device,
        dtype=torch.float32,
    )
    info = dict(last_info())
    legalized = {
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
    geometry = target_free_geometry_metrics(
        legalized_boxes,
        batch["areas"][row, :count],
        batch["b2b"][row, : int(batch["n_b2b"][row])],
        batch["p2b"][row, : int(batch["n_p2b"][row])],
        batch["pins"][row, : int(batch["n_pins"][row])],
        int(batch["n_b2b"][row]),
        int(batch["n_p2b"][row]),
    )
    return {
        "name": name,
        "legalized_xywh": legalized_boxes.cpu().tolist(),
        "stages": {
            "refined": stage_metrics(refined_boxes_row, batch, row),
            "legalized": legalized,
        },
        "displacement": displacement(
            refined_boxes_row[None],
            legalized_boxes[None],
            constraints[None],
            scale[None],
        ),
        "legalizer": info,
        "target_free_geometry": geometry,
    }


def view(name: str, row: dict) -> LegalizerCandidate:
    legalized = row["stages"]["legalized"]
    return LegalizerCandidate(
        name=name,
        v_rel=float(legalized["v_rel"]),
        displacement=float(row["displacement"]["center_l2"]),
        boundary=float(legalized["boundary"]),
        grouping=float(legalized["grouping"]),
        rung=int(row["legalizer"]["rung"]),
        solves=int(row["legalizer"]["solves"]),
        b2b_hpwl=float(row["target_free_geometry"]["b2b_hpwl"]),
        p2b_hpwl=float(row["target_free_geometry"]["p2b_hpwl"]),
        outline_compactness=float(
            row["target_free_geometry"]["outline_compactness"]
        ),
        outline_aspect=float(row["target_free_geometry"]["outline_aspect"]),
    )


def paired(selected: list[dict], baseline: list[dict], metric: str) -> dict:
    current = [
        {
            "index": row["index"],
            "value": float(row["selected"]["stages"]["legalized"][metric]),
        }
        for row in selected
    ]
    reference = [
        {
            "index": row["index"],
            "value": float(row["baseline"]["stages"]["legalized"][metric]),
        }
        for row in baseline
    ]
    return paired_delta(
        current,
        reference,
        "value",
        iterations=10000,
        lower_is_better=True,
    )


def main() -> None:
    args = parse_args()
    if args.single_parent_reflow and not args.legal_parent_reflow:
        raise ValueError("--single-parent-reflow requires --legal-parent-reflow")
    if args.deterministic:
        enable_deterministic_execution()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = load_chip_checkpoint(checkpoint, device, use_ema=True)
    store = PackedCaseStore(args.cache)
    indices = load_indices(args.case_indices) if args.case_indices else store.indices().tolist()
    weights = parse_refiner_weight_overrides(args.refiner_weight)
    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for count in sorted({int(store.arrays["n"][index]) for index in indices}):
            if args.blocks is not None and count not in args.blocks:
                continue
            group = [index for index in indices if int(store.arrays["n"][index]) == count]
            for start in range(0, len(group), args.batch_size):
                case_indices = group[start : start + args.batch_size]
                batch = store.batch(np.asarray(case_indices), device=device)
                batch["_indices"] = torch.as_tensor(case_indices, device=device)
                routing_features = router_features(
                    batch["constraints"],
                    batch["b2b"],
                    batch["p2b"],
                    batch["n_b2b"],
                    batch["n_p2b"],
                    batch["areas"],
                    batch["pins"],
                    batch["n_pins"],
                ).cpu()
                refine_case = build_refine_case(make_instances(batch), 1, device)
                trajectories = {}
                projector_grid = (8,) if args.single_parent_reflow else (8, 32)
                for projector_iterations in projector_grid:
                    configured = configure_chip_projector(
                        model,
                        enabled=True,
                        iterations=projector_iterations,
                        boundary_axes=False,
                        boundary_lock_corners=True,
                    )
                    generator = torch.Generator(device=device).manual_seed(
                        stable_batch_seed(args.seed, case_indices)
                    )
                    raw, _ = sample_for_batch(
                        configured,
                        batch,
                        sampling_steps=args.nfe,
                        generator=generator,
                    )
                    raw_boxes, _ = aggregate_physics(raw, batch)
                    refined = refine_closed(
                        raw,
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
                    refined_boxes, _ = aggregate_physics(refined, batch)
                    trajectories[projector_iterations] = (
                        refined,
                        refined_boxes,
                        raw_boxes,
                    )
                batch_rows = []
                for row, case_index in enumerate(case_indices):
                    constraints = batch["constraints"][row, :count]
                    scale = batch["areas"][row, :count].sum().sqrt().clamp_min(1.0e-6)
                    refined8, boxes8, raw8 = trajectories[8]
                    baseline = legalized_candidate(
                        "baseline",
                        refined8,
                        boxes8,
                        batch,
                        row,
                        constraints,
                        scale,
                        relation="restored",
                        time_limit_ms=args.legalizer_time_limit_ms,
                        device=device,
                    )
                    expanded = legalized_candidate(
                        "expanded",
                        refined8,
                        boxes8,
                        batch,
                        row,
                        constraints,
                        scale,
                        relation="expanded",
                        time_limit_ms=args.legalizer_time_limit_ms,
                        device=device,
                    )
                    candidate_entries = {
                        "baseline": baseline,
                        "expanded": expanded,
                    }
                    candidate_views = [view("expanded", expanded)]
                    raw_stages = {
                        "projector8": stage_metrics(raw8[row, :count], batch, row),
                    }
                    if 32 in trajectories:
                        refined32, boxes32, raw32 = trajectories[32]
                        projector32 = legalized_candidate(
                            "projector32",
                            refined32,
                            boxes32,
                            batch,
                            row,
                            constraints,
                            scale,
                            relation="restored",
                            time_limit_ms=args.legalizer_time_limit_ms,
                            device=device,
                        )
                        candidate_entries["projector32"] = projector32
                        candidate_views.append(view("projector32", projector32))
                        raw_stages["projector32"] = stage_metrics(
                            raw32[row, :count],
                            batch,
                            row,
                        )
                    choice = select_legalizer_candidate(
                        view("baseline", baseline),
                        tuple(candidate_views),
                        policy=args.selection_policy,
                    )
                    selected = candidate_entries[choice.name]
                    batch_rows.append(
                        {
                            "index": int(case_index),
                            "blocks": count,
                            "selection": choice.name,
                            "router_features": routing_features[row].tolist(),
                            "baseline": baseline,
                            "expanded": expanded,
                            **(
                                {"projector32": candidate_entries["projector32"]}
                                if "projector32" in candidate_entries
                                else {}
                            ),
                            "selected": selected,
                            "raw": raw_stages,
                            "_parent_latent": encode_latents(
                                torch.as_tensor(
                                    selected["legalized_xywh"],
                                    device=device,
                                    dtype=torch.float32,
                                )[None],
                                batch["areas"][row, :count][None],
                            )[0],
                        }
                    )
                if args.legal_parent_reflow:
                    parent_latent = torch.stack(
                        [row["_parent_latent"] for row in batch_rows]
                    )
                    parent_boxes, _ = aggregate_physics(parent_latent, batch)
                    masks = torch.stack(
                        [
                            critical_block_mask(
                                parent_boxes[row, :count],
                                batch["b2b"][row, : int(batch["n_b2b"][row])],
                                batch["p2b"][row, : int(batch["n_p2b"][row])],
                                batch["pins"][row, : int(batch["n_pins"][row])],
                                batch["constraints"][row, :count],
                                int(batch["n_b2b"][row]),
                                int(batch["n_p2b"][row]),
                                top_fraction=args.reflow_top_fraction,
                            )
                            for row in range(len(case_indices))
                        ]
                    )
                    configure_chip_projector(
                        model,
                        enabled=True,
                        iterations=8,
                        boundary_axes=False,
                        boundary_lock_corners=True,
                    )
                    reflow_generator = torch.Generator(device=device).manual_seed(
                        stable_batch_seed(args.seed + 1000, case_indices)
                    )
                    reflow_latent, reflow_diagnostics = partial_renoise_parent(
                        model,
                        batch,
                        parent_latent,
                        masks,
                        reflow_generator,
                        noise_fraction=args.reflow_noise_fraction,
                        steps=args.reflow_steps,
                        projector_iterations=8,
                    )
                    reflow_boxes, _ = aggregate_physics(reflow_latent, batch)
                    for row, row_data in enumerate(batch_rows):
                        row_data.pop("_parent_latent")
                        constraints = batch["constraints"][row, :count]
                        scale = batch["areas"][row, :count].sum().sqrt().clamp_min(
                            1.0e-6
                        )
                        reflow = legalized_candidate(
                            "reflow",
                            reflow_latent,
                            reflow_boxes,
                            batch,
                            row,
                            constraints,
                            scale,
                            relation="restored",
                            time_limit_ms=args.legalizer_time_limit_ms,
                            device=device,
                        )
                        parent_entry = row_data["selected"]
                        parent_name = row_data["selection"]
                        final_choice = select_legalizer_candidate(
                            view(f"parent:{parent_name}", parent_entry),
                            (view("reflow", reflow),),
                            policy=args.reflow_selection_policy,
                        )
                        final_entry = (
                            parent_entry
                            if final_choice.name.startswith("parent:")
                            else reflow
                        )
                        row_data["parent_selected"] = parent_entry
                        row_data["reflow"] = reflow
                        row_data["selected"] = final_entry
                        row_data["selection"] = final_choice.name
                        row_data["reflow_diagnostics"] = {
                            key: float(values if values.ndim == 0 else values[row])
                            for key, values in reflow_diagnostics.items()
                        }
                    rows.extend(batch_rows)
                else:
                    for row_data in batch_rows:
                        row_data.pop("_parent_latent")
                    rows.extend(batch_rows)
    baseline_rows = rows
    selection_names = sorted({row["selection"] for row in rows})
    summary = {
        "cases": len(rows),
        "selection_counts": {
            name: sum(row["selection"] == name for row in rows)
            for name in selection_names
        },
        "solver_escalations": [
            row["index"]
            for row in rows
            if row["selected"]["legalizer"]["rung"]
            > row["baseline"]["legalizer"]["rung"]
            or row["selected"]["legalizer"]["solves"]
            > row["baseline"]["legalizer"]["solves"]
        ],
        "selected_feasible_fraction": float(
            np.mean([row["selected"]["stages"]["legalized"]["feasible"] for row in rows])
        ),
        "paired_cost": paired(rows, baseline_rows, "cost"),
        "paired_v_rel": paired(rows, baseline_rows, "v_rel"),
        "wall_clock_seconds": time.perf_counter() - started,
    }
    report = {
        "version": "chip-m48-portfolio-evaluator-v1",
        "status": (
            "pass"
            if not summary["solver_escalations"]
            and summary["selected_feasible_fraction"] == 1.0
            and summary["paired_cost"]["relative_delta"] < 0.0
            and summary["paired_v_rel"]["relative_delta"] < 0.0
            else "rejected"
        ),
        "selection_contract": {
            "uses_target": False,
            "key": ["v_rel", "legalizer_displacement", "boundary", "grouping", "name"],
            "eligibility": "selected rung/solves <= baseline",
            "policy": args.selection_policy,
            "reflow_policy": (
                args.reflow_selection_policy if args.legal_parent_reflow else None
            ),
        },
        "budget": {
            "nfe": (
                args.nfe + args.reflow_steps
                if args.single_parent_reflow
                else 2 * args.nfe
                + (args.reflow_steps if args.legal_parent_reflow else 0)
            ),
            "refiner_steps": (
                args.refiner_steps
                if args.single_parent_reflow
                else 2 * args.refiner_steps
            ),
            "legalizer_calls": (
                3
                if args.single_parent_reflow
                else 4
                if args.legal_parent_reflow
                else 3
            ),
            "projector_iterations": [8] if args.single_parent_reflow else [8, 32],
            "legal_parent_reflow": bool(args.legal_parent_reflow),
            "single_parent_reflow": bool(args.single_parent_reflow),
            "reflow_steps": args.reflow_steps if args.legal_parent_reflow else 0,
        },
        "summary": summary,
        "rows": rows,
        "boundary": "Bounded portfolio result; doubled generation budget is mandatory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

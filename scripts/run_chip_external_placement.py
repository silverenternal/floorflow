#!/usr/bin/env python3
"""Generate and legalize one external ORFS macro placement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import PAPER_WEIGHTS, refine_closed

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.def_graph import parse_def_graph
from floorflow.chip_flow.external_portfolio import (
    decode_refined,
    displacement_summary,
    geometry_audit,
    legalized_candidate,
    refinement_input,
    select_candidate,
)
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.ppa_external import (
    export_macro_tcl,
    floorplan_instance,
    guard_small_graph_pe,
    infer_hierarchy_clusters,
    model_batch,
    parse_orfs_graph,
    physical_boxes,
    placement_connectivity,
)
from floorflow.chip_flow.refiner_config import parse_refiner_weight_overrides
from floorflow.chip_flow.reproducibility import enable_deterministic_execution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--odb", type=Path)
    parser.add_argument("--def", dest="def_path", type=Path)
    parser.add_argument("--macro-lef", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--refiner-steps", type=int, default=400)
    parser.add_argument("--refiner-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--refiner-schedule",
        choices=("linear", "cosine", "constant"),
        default="linear",
    )
    parser.add_argument("--refiner-weight", action="append", default=[])
    parser.add_argument(
        "--refiner-shape",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--refiner-optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--refiner-chunk", type=int, default=5)
    parser.add_argument("--projector-iterations", type=int, default=8)
    parser.add_argument("--portfolio", action="store_true")
    parser.add_argument("--portfolio-projector-iterations", type=int, default=32)
    parser.add_argument("--projector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-shelf", action="store_true")
    parser.add_argument("--position-aware-shelf", action="store_true")
    parser.add_argument("--macro-halo-um", type=float, default=0.0)
    parser.add_argument("--protocol-tag", default="")
    parser.add_argument("--legalizer-time-limit-ms", type=int, default=2000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--instance-hierarchy", action="store_true")
    parser.add_argument("--preserve-reference-orientation", action="store_true")
    parser.add_argument(
        "--legalizer-relation",
        choices=("restored", "expanded"),
        default="restored",
    )
    parser.add_argument(
        "--orfs-image",
        default="openroad/orfs:chips-cvpr-703649ba",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_def_adapter_cpu_v1.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_graph(args: argparse.Namespace) -> str:
    relative = args.odb.resolve().relative_to(ROOT)
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/work",
        "--entrypoint",
        "/bin/bash",
        args.orfs_image,
        "-lc",
        (
            "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad "
            "-exit -no_init -no_splash -threads 1 "
            f"-db /work/{relative} /work/scripts/orfs_export_macro_graph.tcl"
        ),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def load_graph(args: argparse.Namespace):
    if args.def_path is not None:
        if not args.macro_lef:
            raise ValueError("--def requires at least one --macro-lef")
        return parse_def_graph(args.def_path, args.macro_lef), "def_lef"
    if args.odb is None:
        raise ValueError("one of --odb or --def is required")
    return parse_orfs_graph(export_graph(args)), "odb"


def generate_trajectory(
    model,
    batch: dict,
    graph,
    instance,
    refine_case,
    weights: dict[str, float],
    args: argparse.Namespace,
    projector_iterations: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    model.projector = bool(args.projector)
    model.projector_iterations = int(projector_iterations)
    generator = torch.Generator(device=args.device).manual_seed(int(args.seed))
    latent, diagnostics = sample_for_batch(
        model,
        batch,
        sampling_steps=int(args.nfe),
        generator=generator,
    )
    model_boxes = physical_boxes(latent, graph)
    z0 = refinement_input(latent, graph, instance)
    refined_latent = refine_closed(
        z0,
        refine_case,
        weights,
        int(args.refiner_steps),
        args.refiner_lr,
        args.refiner_schedule,
        chunk=args.refiner_chunk,
        compile_loop=False,
        shape=args.refiner_shape,
        optimizer=args.refiner_optimizer,
    )[int(args.refiner_steps)]
    return (
        model_boxes,
        decode_refined(refined_latent, graph, instance),
        diagnostics,
    )




def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.deterministic:
        enable_deterministic_execution()
    graph, graph_source = load_graph(args)
    device = torch.device(args.device)
    cluster_ids = infer_hierarchy_clusters(graph) if args.instance_hierarchy else None
    batch = model_batch(graph, device, cluster_ids=cluster_ids)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = load_chip_checkpoint(checkpoint, device, use_ema=True)
    small_graph_pe_fallback = guard_small_graph_pe(model, int(batch["n"]))
    instance = floorplan_instance(graph, cluster_ids=cluster_ids)
    refine_case = build_refine_case([instance], 1, device)
    base_weights = dict(PAPER_WEIGHTS)
    base_weights["outline"] = 1.0
    weights = parse_refiner_weight_overrides(
        args.refiner_weight,
        base=base_weights,
    )
    model_boxes, refined_boxes, diagnostics = generate_trajectory(
        model,
        batch,
        graph,
        instance,
        refine_case,
        weights,
        args,
        args.projector_iterations,
    )
    legalization_instance = floorplan_instance(
        graph,
        halo_um=args.macro_halo_um,
        cluster_ids=cluster_ids,
    )
    baseline = legalized_candidate(
        "baseline",
        refined_boxes,
        graph,
        instance,
        legalization_instance,
        relation=args.legalizer_relation,
        halo_um=args.macro_halo_um,
        force_shelf=args.force_shelf,
        position_aware_shelf=args.position_aware_shelf,
        time_limit_ms=args.legalizer_time_limit_ms,
    )
    candidates = (baseline,)
    portfolio_rows = None
    selected = baseline
    if args.portfolio:
        alt_model_boxes, alt_refined, _ = generate_trajectory(
            model,
            batch,
            graph,
            instance,
            refine_case,
            weights,
            args,
            args.portfolio_projector_iterations,
        )
        expanded = legalized_candidate(
            "expanded",
            refined_boxes,
            graph,
            instance,
            legalization_instance,
            relation="expanded",
            halo_um=args.macro_halo_um,
            force_shelf=args.force_shelf,
            position_aware_shelf=args.position_aware_shelf,
            time_limit_ms=args.legalizer_time_limit_ms,
        )
        alt = legalized_candidate(
            "projector32",
            alt_refined,
            graph,
            instance,
            legalization_instance,
            relation=args.legalizer_relation,
            halo_um=args.macro_halo_um,
            force_shelf=args.force_shelf,
            position_aware_shelf=args.position_aware_shelf,
            time_limit_ms=args.legalizer_time_limit_ms,
        )
        candidates += (expanded, alt)
        selected = select_candidate(baseline, (expanded, alt))
        portfolio_rows = {
            "selection": selected.name,
            "candidates": [
                {
                    "name": candidate.name,
                    "route": candidate.route,
                    "v_rel": candidate.v_rel,
                    "boundary": candidate.boundary,
                    "grouping": candidate.grouping,
                    "displacement_mean_um": candidate.displacement_mean_um,
                    "rung": candidate.rung,
                    "solves": candidate.solves,
                    "eligible": candidate.eligible,
                }
                for candidate in candidates
            ],
            "alt_model_boxes": alt_model_boxes.tolist(),
            "alt_refined_boxes": alt_refined.tolist(),
        }
    legalized_boxes = selected.boxes
    legalizer_route = selected.route
    rows = {
        "model_output": geometry_audit(model_boxes, graph, instance),
        "refined_output": geometry_audit(refined_boxes, graph, instance),
        "legalized_output": geometry_audit(legalized_boxes, graph, instance),
        "scale_pack_intermediate": selected.geometry,
        "scale_pack_translated": None,
    }
    placements_name = (
        f"placements_{args.protocol_tag}"
        if args.protocol_tag
        else "placements"
    )
    output_stem = (
        ROOT
        / "outputs/chip_ppa"
        / placements_name
        / args.case
        / f"{args.arm}_seed{args.seed}"
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    tcl_path = output_stem.with_suffix(".tcl")
    boxes_path = output_stem.with_suffix(".json")
    trace_path = output_stem.with_suffix(".trace.json")
    tcl_path.write_text(
        export_macro_tcl(
            graph,
            legalized_boxes,
            preserve_reference_orientation=args.preserve_reference_orientation,
        ),
        encoding="utf-8",
    )
    boxes_path.write_text(
        json.dumps(
            {
                "case": args.case,
                "arm": args.arm,
                "seed": args.seed,
                "checkpoint": str(args.checkpoint),
                "macros": [
                    {
                        "name": macro["name"],
                        "master": macro["master"],
                        "x_um": float(box[0]),
                        "y_um": float(box[1]),
                        "width_um": float(box[2]),
                        "height_um": float(box[3]),
                        "orientation": (
                            macro.get("orientation", "R0")
                            if args.preserve_reference_orientation
                            else "R0"
                        ),
                    }
                    for macro, box in zip(graph.macros, legalized_boxes)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path.write_text(
        json.dumps(
            {
                "case": args.case,
                "arm": args.arm,
                "seed": args.seed,
                "macro_names": [macro["name"] for macro in graph.macros],
                "model_boxes": model_boxes.tolist(),
                "refined_boxes": refined_boxes.tolist(),
                "legalized_boxes": legalized_boxes.tolist(),
                "portfolio": portfolio_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checks = {
        "macro_count_matches_export": len(legalized_boxes) == len(graph.macros),
        "legalized_checker_feasible": rows["legalized_output"]["checker_feasible"],
        "legalized_inside_core": rows["legalized_output"]["inside_core"],
        "legalized_zero_overlap": rows["legalized_output"]["overlap_pair_count"] == 0,
        "fixed_shapes_preserved": rows["legalized_output"][
            "fixed_shape_max_relative_error"
        ]
        < 1.0e-6,
    }
    report = {
        "version": "2026-10-31-chip-def-adapter-portfolio-v2",
        "status": "pass" if all(checks.values()) else "fail",
        "case": args.case,
        "arm": args.arm,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "device": str(device),
        "nfe": int(args.nfe),
        "refiner_steps": int(args.refiner_steps),
        "projector_iterations": int(args.projector_iterations),
        "portfolio": bool(args.portfolio),
        "portfolio_projector_iterations": int(
            args.portfolio_projector_iterations
        ),
        "portfolio_selection": selected.name,
        "portfolio_candidates": (
            portfolio_rows["candidates"] if portfolio_rows is not None else None
        ),
        "projector_enabled": bool(args.projector),
        "force_shelf": bool(args.force_shelf),
        "macro_halo_um": float(args.macro_halo_um),
        "instance_hierarchy": bool(args.instance_hierarchy),
        "preserve_reference_orientation": bool(
            args.preserve_reference_orientation
        ),
        "inferred_cluster_count": max(cluster_ids.values(), default=0) if cluster_ids else 0,
        "legalizer_relation": args.legalizer_relation,
        "protocol_tag": args.protocol_tag,
        "legalizer_route": legalizer_route,
        "refiner": {
            "steps": args.refiner_steps,
            "weights": weights,
            "learning_rate": args.refiner_lr,
            "schedule": args.refiner_schedule,
            "shape": args.refiner_shape,
            "optimizer": args.refiner_optimizer,
        },
        "legalizer_time_limit_ms": args.legalizer_time_limit_ms,
        "legalizer_glop_threads": 1,
        "deterministic": args.deterministic,
        "scale_pack_last_info": {
            "rung": selected.rung,
            "solves": selected.solves,
        },
        "small_graph_pe_fallback": small_graph_pe_fallback,
        "graph": {
            "source": graph_source,
            "macros": len(graph.macros),
            "ports": len(graph.ports),
            "b2b_edges": int(batch["n_b2b"].item()),
            "p2b_edges": int(batch["n_p2b"].item()),
        },
        "diagnostics": {
            key: float(value.float().mean().cpu())
            for key, value in diagnostics.items()
        },
        "geometry": rows,
        "connectivity": {
            "model_output": placement_connectivity(model_boxes, graph),
            "refined_output": placement_connectivity(refined_boxes, graph),
            "legalized_output": placement_connectivity(legalized_boxes, graph),
        },
        "legalization_displacement": displacement_summary(
            refined_boxes,
            legalized_boxes,
        ),
        "checks": checks,
        "outputs": {
            "tcl": str(tcl_path.relative_to(ROOT)),
            "boxes": str(boxes_path.relative_to(ROOT)),
            "trace": str(trace_path.relative_to(ROOT)),
            "tcl_sha256": sha256(tcl_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

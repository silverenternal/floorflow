"""Run the frozen S8 DBU packing gate on a small, report-backed case subset."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.scoring.cost import score_full_fast
from trinity.floorplan.types import FloorplanInstance, Placement

from floorflow.chip_flow.dbu_endpoint_portfolio import (
    project_dbu_endpoint_portfolio,
)
from floorflow.chip_flow.dbu_projection import project_quantized_layout
from floorflow.chip_flow.dbu_quantization import quantize_boxes_to_dbu
from floorflow.chip_flow.dbu_shape_quantization import (
    quantize_boxes_adjacency_aware,
)
from floorflow.chip_flow.score_diagnostics import score_diagnostics

DEFAULT_CASES = (57, 231, 254, 676, 681)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/reports/chip_s6_cluster_reflow_actions_val92_quantized_v3.json"),
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("outputs/cohorts/chip_s205_consensus_500_v4"),
    )
    parser.add_argument("--cases", default=",".join(map(str, DEFAULT_CASES)))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resolution", type=float, default=0.001)
    parser.add_argument("--area-tolerance", type=float, default=0.01)
    parser.add_argument("--grid-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--max-displacement-fraction", type=float, default=0.5)
    parser.add_argument("--time-limit-ms", type=int, default=2000)
    parser.add_argument("--retry-time-limit-ms", type=int, default=10000)
    parser.add_argument(
        "--shape-quantizer",
        choices=("standard", "adjacency"),
        default="standard",
    )
    parser.add_argument(
        "--contact-reference",
        choices=("incoming", "legalized"),
        default="incoming",
    )
    parser.add_argument("--endpoint-portfolio", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _case_indices(raw: str, limit: int) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one case index is required")
    return values[:limit] if limit > 0 else values


def _geometry(payload) -> dict[str, np.ndarray]:
    return {
        "gt_boxes": np.asarray(payload["source_gt_boxes"]),
        "areas": np.asarray(payload["source_areas"]),
        "constraints": np.asarray(payload["source_constraints"]),
        "b2b": np.asarray(payload["source_b2b"]),
        "p2b": np.asarray(payload["source_p2b"]),
        "pins": np.asarray(payload["source_pins"]),
    }


def _instance(geometry: dict[str, np.ndarray]) -> FloorplanInstance:
    constraints = geometry["constraints"]
    count = len(geometry["areas"])
    targets = np.full((count, 4), -1.0, dtype=np.float64)
    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    targets[fixed, 2:] = geometry["gt_boxes"][fixed, 2:]
    targets[preplaced] = geometry["gt_boxes"][preplaced]
    return FloorplanInstance(
        block_count=count,
        area_targets=geometry["areas"],
        constraints=constraints,
        b2b=geometry["b2b"],
        p2b=geometry["p2b"],
        pins_pos=geometry["pins"],
        target_positions=targets,
        gt_positions=geometry["gt_boxes"],
    )


def _audit(
    boxes: np.ndarray,
    instance: FloorplanInstance,
    geometry: dict[str, np.ndarray],
    resolution: float,
) -> tuple[dict, dict]:
    score = score_full_fast(
        Placement(
            xywh=np.asarray(boxes, dtype=np.float64).copy(),
            instance=copy.deepcopy(instance),
        )
    )
    diagnostics = score_diagnostics(
        "candidate",
        score,
        boxes,
        geometry["gt_boxes"],
        geometry["b2b"],
        geometry["p2b"],
        geometry["pins"],
    )
    report = score.feasibility
    soft = score.soft
    aligned = np.rint(np.asarray(boxes) / resolution) * resolution
    summary = {
        "hard_cost": float(score.cost),
        "v_rel": float(score.v_rel),
        "overlap_pairs": int(report.overlap_count),
        "area_violations": len(report.area_violations),
        "fixed_violations": len(report.fixed_violations),
        "preplaced_violations": len(report.preplaced_violations),
        "aspect_violations": len(report.aspect_violations),
        "hard_feasible": bool(report.is_feasible()),
        "grouping": int(soft.grouping) if soft is not None else 0,
        "mib": int(soft.mib) if soft is not None else 0,
        "boundary": int(soft.boundary) if soft is not None else 0,
        "max_grid_residual": float(np.max(np.abs(np.asarray(boxes) - aligned), initial=0.0)),
        "diagnostics": {key.removeprefix("candidate_"): value for key, value in diagnostics.items()},
    }
    return summary, diagnostics


def _quantize(
    boxes: np.ndarray,
    geometry: dict[str, np.ndarray],
    *,
    resolution: float,
    area_tolerance: float,
):
    return quantize_boxes_to_dbu(
        torch.from_numpy(np.asarray(boxes, dtype=np.float32))[None],
        torch.from_numpy(np.asarray(geometry["areas"], dtype=np.float32))[None],
        torch.from_numpy(np.asarray(geometry["constraints"], dtype=np.float32))[None],
        resolution=resolution,
        area_tolerance=area_tolerance,
    )


def _strict_gate(
    reference: dict,
    repaired: dict,
    *,
    accepted: bool,
    max_displacement: float,
    span: float,
    grid_tolerance: float,
    max_displacement_fraction: float,
) -> dict:
    checks = {
        "route_accepted": bool(accepted),
        "hard_feasible": bool(repaired["hard_feasible"]),
        "area_zero": repaired["area_violations"] == 0,
        "fixed_zero": repaired["fixed_violations"] == 0,
        "preplaced_zero": repaired["preplaced_violations"] == 0,
        "aspect_zero": repaired["aspect_violations"] == 0,
        "grid_aligned": repaired["max_grid_residual"] <= grid_tolerance,
        "grouping_not_worse": repaired["grouping"] <= reference["grouping"],
        "boundary_not_worse": repaired["boundary"] <= reference["boundary"],
        "displacement_bounded": (max_displacement <= max_displacement_fraction * max(span, 1.0)),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _layout_span(boxes: np.ndarray) -> float:
    boxes = np.asarray(boxes)
    return max(
        float((boxes[:, 0] + boxes[:, 2]).max() - boxes[:, 0].min()),
        float((boxes[:, 1] + boxes[:, 3]).max() - boxes[:, 1].min()),
        1.0,
    )


def _evaluate_case(
    row: dict,
    cohort_dir: Path,
    args: argparse.Namespace,
) -> dict:
    index = int(row["case_index"])
    path = cohort_dir / f"case_{index:08d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing cohort shard for case {index}: {path}")
    payload = np.load(path)
    if int(payload["case_index"]) != index:
        raise ValueError(f"case-index mismatch in {path}")
    geometry = _geometry(payload)
    instance = _instance(geometry)
    raw = np.asarray(row["legalized_xywh"], dtype=np.float64)
    if raw.shape != (instance.block_count, 4):
        raise ValueError(f"legalized boxes have the wrong shape for case {index}")

    reference, _ = _audit(raw, instance, geometry, args.resolution)
    if args.shape_quantizer == "adjacency":
        shape = quantize_boxes_adjacency_aware(
            torch.from_numpy(raw)[None],
            torch.from_numpy(geometry["areas"].astype(np.float32))[None],
            torch.from_numpy(geometry["constraints"].astype(np.float32))[None],
            instance,
            resolution=args.resolution,
            area_tolerance=args.area_tolerance,
        )
        quantized_boxes = shape.boxes
        quantization = {
            "max_area_relative_error": float(shape.area_relative_error.max()),
            "failing_block_fraction": float(shape.area_violation_mask.mean()),
            "max_position_grid_residual": float(shape.max_grid_residual),
            "max_soft_shape_grid_residual": float(shape.max_grid_residual),
            "fixed_shape_error": float(shape.fixed_shape_error),
            "preplaced_max_error": float(shape.preplaced_max_error),
            "mib_shape_mismatch_count": 0,
            "modified_shapes": int(shape.modified_shapes),
            "translated_components": int(shape.translated_components),
        }
    else:
        quantized = _quantize(
            raw,
            geometry,
            resolution=args.resolution,
            area_tolerance=args.area_tolerance,
        )
        quantized_boxes = (
            quantized.boxes[0].detach().cpu().numpy().astype(np.float64)
        )
        quantization = {
            "max_area_relative_error": float(quantized.max_area_relative_error),
            "failing_block_fraction": float(quantized.failing_block_fraction),
            "max_position_grid_residual": float(
                quantized.max_position_grid_residual
            ),
            "max_soft_shape_grid_residual": float(
                quantized.max_soft_shape_grid_residual
            ),
            "fixed_shape_error": float(quantized.fixed_shape_error),
            "preplaced_max_error": float(quantized.preplaced_max_error),
            "mib_shape_mismatch_count": int(quantized.mib_shape_mismatch_count),
        }
    quantized_reference, _ = _audit(
        quantized_boxes,
        instance,
        geometry,
        args.resolution,
    )
    started = time.perf_counter()
    if args.endpoint_portfolio:
        portfolio = project_dbu_endpoint_portfolio(
            quantized_boxes,
            instance,
            reference_boxes=raw,
            resolution=args.resolution,
            legalization_time_limit_ms=args.time_limit_ms,
            projection_time_limit_ms=args.time_limit_ms,
            displacement_p95_fraction=args.max_displacement_fraction,
        )
        repair = portfolio.selected.result
        gate_displacement = portfolio.selected.displacement_p95
        portfolio_payload = {
            "selected": portfolio.selected.name,
            "tried_routes": list(portfolio.tried_routes),
            "candidates": [
                {
                    "name": candidate.name,
                    "accepted": candidate.result.accepted,
                    "route": candidate.result.route,
                    "overlap_after": candidate.result.overlap_after,
                    "p50": candidate.displacement_p50,
                    "p95": candidate.displacement_p95,
                    "max": candidate.max_displacement,
                }
                for candidate in portfolio.candidates
            ],
        }
    else:
        repair = project_quantized_layout(
            quantized_boxes,
            instance,
            resolution=args.resolution,
            legalization_time_limit_ms=args.time_limit_ms,
            projection_time_limit_ms=args.time_limit_ms,
            contact_reference=raw if args.contact_reference == "legalized" else None,
        )
        gate_displacement = repair.max_displacement
        portfolio_payload = None
    elapsed = time.perf_counter() - started
    if repair.continuous_boxes is None:
        raise RuntimeError(f"projection did not expose a continuous reference: {index}")
    projection_reference, _ = _audit(
        repair.continuous_boxes,
        instance,
        geometry,
        args.resolution,
    )
    repaired, _ = _audit(
        repair.boxes,
        instance,
        geometry,
        args.resolution,
    )
    gate = _strict_gate(
        projection_reference,
        repaired,
        accepted=repair.accepted,
        max_displacement=gate_displacement,
        span=_layout_span(quantized_boxes),
        grid_tolerance=args.grid_tolerance,
        max_displacement_fraction=args.max_displacement_fraction,
    )
    raw_gate = _strict_gate(
        reference,
        repaired,
        accepted=repair.accepted,
        max_displacement=gate_displacement,
        span=_layout_span(quantized_boxes),
        grid_tolerance=args.grid_tolerance,
        max_displacement_fraction=args.max_displacement_fraction,
    )
    return {
        "case_index": index,
        "blocks": instance.block_count,
        "cohort_shard": str(path),
        "quantization": quantization,
        "reference": reference,
        "quantized_reference": quantized_reference,
        "projection_reference": projection_reference,
        "repaired": repaired,
        "repair": {
            "accepted": bool(repair.accepted),
            "route": repair.route,
            "solver": repair.solver,
            "overlap_before": int(repair.overlap_before),
            "overlap_after": int(repair.overlap_after),
            "area_violations": int(repair.area_violations),
            "fixed_violations": int(repair.fixed_violations),
            "preplaced_violations": int(repair.preplaced_violations),
            "grouping_before": int(repair.grouping_before),
            "grouping_after": int(repair.grouping_after),
            "boundary_before": int(repair.boundary_before),
            "boundary_after": int(repair.boundary_after),
            "max_grid_residual": float(repair.max_grid_residual),
            "max_displacement": float(repair.max_displacement),
            "moved_blocks": int(repair.moved_blocks),
            "contact_edges": int(repair.contact_edges),
            "solver_seconds": float(repair.solver_seconds),
            "elapsed_seconds": float(elapsed),
        },
        "endpoint_portfolio": portfolio_payload,
        "gate": gate,
        "raw_gate": raw_gate,
    }


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    by_index = {int(row["case_index"]): row for row in report["cases"]}
    case_indices = _case_indices(args.cases, args.limit)
    missing = [index for index in case_indices if index not in by_index]
    if missing:
        raise ValueError(f"report is missing requested cases: {missing}")
    rows = [_evaluate_case(by_index[index], args.cohort_dir, args) for index in case_indices]
    initial_rows = list(rows)
    initial_failed = sum(not bool(row["gate"]["passed"]) for row in rows)
    retried_cases = []
    if args.retry_time_limit_ms > args.time_limit_ms:
        retry_args = argparse.Namespace(**vars(args))
        retry_args.time_limit_ms = args.retry_time_limit_ms
        for position, row in enumerate(rows):
            if row["gate"]["passed"]:
                continue
            index = int(row["case_index"])
            retried_cases.append(index)
            initial_row = row
            rows[position] = _evaluate_case(by_index[index], args.cohort_dir, retry_args)
            rows[position]["initial_repair"] = initial_row["repair"]
            rows[position]["retry_time_limit_ms"] = args.retry_time_limit_ms
    passed = sum(bool(row["gate"]["passed"]) for row in rows)
    raw_passed = sum(bool(row["raw_gate"]["passed"]) for row in rows)
    if passed == len(rows) and raw_passed == len(rows):
        status = "pass"
    elif passed == len(rows):
        status = "quantized_legal_pass_raw_soft_gap_open"
    else:
        status = "fail"
    payload = {
        "version": "chip-s8-dbu-packing-gate-v1",
        "status": status,
        "protocol": {
            "scope": "five-case CPU fail-fast gate",
            "formal_result": False,
            "requires_expansion": "92-case only after this gate passes",
            "report": str(args.report),
            "cohort_dir": str(args.cohort_dir),
            "cases": case_indices,
            "resolution_um": args.resolution,
            "area_tolerance": args.area_tolerance,
            "grid_tolerance": args.grid_tolerance,
            "max_displacement_fraction": args.max_displacement_fraction,
            "legalizer_time_limit_ms": args.time_limit_ms,
            "retry_time_limit_ms": args.retry_time_limit_ms,
            "retry_policy": "retry only initially failed cases with the same algorithm",
            "shape_quantizer": args.shape_quantizer,
            "contact_reference": args.contact_reference,
            "endpoint_portfolio": bool(args.endpoint_portfolio),
            "score_instance": "original instance; quantized instance only for packing",
            "primary_reference": "scale_pack continuous layout on quantized shapes",
            "raw_reference": "unquantized legalized report row",
            "raw_soft_gap": ("reported separately until shape quantization is adjacency-aware"),
        },
        "summary": {
            "cases": len(rows),
            "initial_failed": initial_failed,
            "retried_cases": retried_cases,
            "passed": passed,
            "failed": len(rows) - passed,
            "pass_rate": passed / len(rows),
            "raw_reference_passed": raw_passed,
            "raw_reference_pass_rate": raw_passed / len(rows),
            "total_elapsed_seconds": (
                sum(row["repair"]["elapsed_seconds"] for row in initial_rows)
                + sum(
                    row["repair"]["elapsed_seconds"]
                    for row in rows
                    if "retry_time_limit_ms" in row
                )
            ),
            "final_elapsed_seconds": sum(
                row["repair"]["elapsed_seconds"] for row in rows
            ),
            "initial_elapsed_seconds": sum(
                row["repair"]["elapsed_seconds"] for row in initial_rows
            ),
            "retry_elapsed_seconds": sum(
                row["repair"]["elapsed_seconds"] for row in rows
                if "retry_time_limit_ms" in row
            ),
            "total_solver_seconds": (
                sum(row["repair"]["solver_seconds"] for row in initial_rows)
                + sum(
                    row["repair"]["solver_seconds"]
                    for row in rows
                    if "retry_time_limit_ms" in row
                )
            ),
            "final_solver_seconds": sum(
                row["repair"]["solver_seconds"] for row in rows
            ),
            "initial_solver_seconds": sum(
                row["repair"]["solver_seconds"] for row in initial_rows
            ),
            "retry_solver_seconds": sum(
                row["repair"]["solver_seconds"] for row in rows
                if "retry_time_limit_ms" in row
            ),
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Audit adjacency-aware DBU shape quantization on report-backed cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.scoring.feasibility import check_feasibility
from trinity.floorplan.scoring.soft import check_soft
from trinity.floorplan.types import Placement

from floorflow.chip_flow.dbu_shape_quantization import (
    quantize_boxes_adjacency_aware,
)
from scripts.evaluate_chip_dbu_packing_gate import _geometry, _instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=float, default=0.001)
    parser.add_argument("--area-tolerance", type=float, default=0.01)
    parser.add_argument("--contact-tolerance", type=float, default=0.02)
    return parser.parse_args()


def _case_indices(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = {int(row["case_index"]): row for row in report["cases"]}
    cases = []
    for index in _case_indices(args.cases):
        path = args.cohort_dir / f"case_{index:08d}.npz"
        payload = np.load(path)
        geometry = _geometry(payload)
        instance = _instance(geometry)
        raw = np.asarray(rows[index]["legalized_xywh"], dtype=np.float64)
        result = quantize_boxes_adjacency_aware(
            torch.from_numpy(raw)[None],
            torch.from_numpy(geometry["areas"].astype(np.float32))[None],
            torch.from_numpy(geometry["constraints"].astype(np.float32))[None],
            instance,
            resolution=args.resolution,
            area_tolerance=args.area_tolerance,
            contact_tolerance=args.contact_tolerance,
        )
        reference = check_soft(Placement(raw, instance))
        feasibility = check_feasibility(Placement(result.boxes, instance))
        cases.append(
            {
                "case_index": index,
                "raw_grouping": int(reference.grouping),
                "raw_boundary": int(reference.boundary),
                "quantized_grouping": int(result.grouping_after),
                "quantized_boundary": int(result.boundary_after),
                "grouping_preserved": result.grouping_after <= reference.grouping,
                "boundary_preserved": result.boundary_after <= reference.boundary,
                "boundary_delta": int(result.boundary_after - reference.boundary),
                "overlap_pairs": len(feasibility.overlap_pairs),
                "area_violation_count": int(result.area_violation_mask.sum()),
                "fixed_violations": len(feasibility.fixed_violations),
                "preplaced_violations": len(feasibility.preplaced_violations),
                "max_soft_grid_residual": float(result.max_grid_residual),
                "modified_shapes": int(result.modified_shapes),
                "translated_components": int(result.translated_components),
            }
        )
    grouping = sum(row["grouping_preserved"] for row in cases)
    boundary = sum(row["boundary_preserved"] for row in cases)
    payload = {
        "version": "chip-s8b-shape-quantization-probe-v1",
        "status": (
            "raw_soft_preserved"
            if grouping == len(cases) and boundary == len(cases)
            else "partial_raw_boundary_gap"
        ),
        "scope": "frozen report-backed CPU subset; not a benchmark result",
        "cases": cases,
        "summary": {
            "count": len(cases),
            "grouping_preserved": grouping,
            "boundary_preserved": boundary,
            "area_violation_count": sum(row["area_violation_count"] for row in cases),
            "fixed_violation_count": sum(row["fixed_violations"] for row in cases),
            "preplaced_violation_count": sum(
                row["preplaced_violations"] for row in cases
            ),
            "max_soft_grid_residual": max(
                row["max_soft_grid_residual"] for row in cases
            ),
        },
        "protocol": {
            "resolution_um": args.resolution,
            "area_tolerance": args.area_tolerance,
            "contact_tolerance_um": args.contact_tolerance,
            "score_instance": "original instance",
            "stage": "shape/contact representation before hard legalization",
            "overlap_pairs_are_expected": True,
            "tested_cases": _case_indices(args.cases),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()

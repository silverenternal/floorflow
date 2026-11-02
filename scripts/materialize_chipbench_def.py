#!/usr/bin/env python3
"""Materialize one candidate placement JSON into a macro-placed DEF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.chip_flow.def_graph import parse_def_graph
from floorflow.chip_flow.def_materialize import (
    load_macro_boxes,
    materialize_macro_def,
)
from floorflow.chip_flow.def_pin_alignment import align_boxes_to_reference

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-def", type=Path, required=True)
    parser.add_argument("--candidate-boxes", type=Path, required=True)
    parser.add_argument("--output-def", type=Path, required=True)
    parser.add_argument("--dbu-per-micron", type=int, default=2000)
    parser.add_argument("--coordinate-grid-dbu", type=int, default=1)
    parser.add_argument("--reference-def", type=Path)
    parser.add_argument("--pin-grid-dbu", type=int, default=280)
    parser.add_argument(
        "--pin-rollback-conflicts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--remove-blockages", action="store_true")
    parser.add_argument("--verify-macro-lef", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def local(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    boxes = load_macro_boxes(local(args.candidate_boxes))
    if args.coordinate_grid_dbu <= 0:
        raise ValueError("coordinate grid must be positive")
    if args.coordinate_grid_dbu > 1:
        for row in boxes.values():
            for key in ("x_um", "y_um"):
                units = float(row[key]) * args.dbu_per_micron
                units = round(units / args.coordinate_grid_dbu)
                row[key] = units * args.coordinate_grid_dbu / args.dbu_per_micron
    pin_alignment = None
    if args.reference_def is not None:
        boxes, pin_alignment = align_boxes_to_reference(
            boxes,
            local(args.reference_def),
            grid_dbu=args.pin_grid_dbu,
            dbu_per_micron=args.dbu_per_micron,
            rollback_conflicts=args.pin_rollback_conflicts,
        )
    output_def = local(args.output_def)
    result = materialize_macro_def(
        local(args.source_def),
        output_def,
        boxes,
        dbu_per_micron=args.dbu_per_micron,
        remove_blockages=args.remove_blockages,
    )
    result["roundtrip"] = None
    result["coordinate_grid_dbu"] = args.coordinate_grid_dbu
    result["pin_alignment"] = pin_alignment
    if args.verify_macro_lef:
        graph = parse_def_graph(
            output_def,
            [local(path) for path in args.verify_macro_lef],
        )
        actual = {macro["name"]: macro for macro in graph.macros}
        errors = []
        for name, row in boxes.items():
            macro = actual.get(name)
            if macro is None:
                errors.append(float("inf"))
                continue
            errors.extend(
                [
                    abs(
                        macro["x_dbu"] / args.dbu_per_micron
                        - float(row["x_um"])
                    ),
                    abs(
                        macro["y_dbu"] / args.dbu_per_micron
                        - float(row["y_um"])
                    ),
                ]
            )
        maximum = max(errors, default=0.0)
        result["roundtrip"] = {
            "status": "pass"
            if len(actual) == len(boxes) and maximum <= 0.001
            else "fail",
            "expected_macros": len(boxes),
            "materialized_macros": len(actual),
            "ports": len(graph.ports),
            "max_coordinate_error_um": maximum,
        }
    if result["roundtrip"] is not None and result["roundtrip"]["status"] != "pass":
        raise SystemExit(1)
    if args.report is not None:
        report = local(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

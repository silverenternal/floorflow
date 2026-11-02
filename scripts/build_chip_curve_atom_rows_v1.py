#!/usr/bin/env python3
"""Normalize the M27 checkpoint-refiner curve into atom metric rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.chip_flow.training_runtime import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve",
        type=Path,
        default=Path(
            "outputs/reports/chip_m27_canonical_refiner_curve_v1_cases.json"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "outputs/reports/chip_m27_canonical_refiner_curve_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/chip_curve_atom_rows_v1.json"),
    )
    args = parser.parse_args()
    curve = json.loads(args.curve.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    seed = int(summary["seed"])
    rows = []
    for checkpoint, checkpoint_rows in curve.items():
        if not isinstance(checkpoint_rows, list):
            raise TypeError(f"{checkpoint} must contain a row list")
        for row in checkpoint_rows:
            rows.append(
                {
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "case_id": str(row["index"]),
                    "proxy_metrics": {
                        "raw_grouping_violations": float(row["raw_grouping"]),
                        "raw_mib_violations": float(row["raw_mib"]),
                        "raw_boundary_violations": float(row["raw_boundary"]),
                        "raw_aspect_violations": float(
                            row["raw_aspect_violations"]
                        ),
                        "raw_overlap_pairs": float(row["raw_overlap_pairs"]),
                        "legalizer_rung": float(row["legalizer_rung"]),
                        "legalizer_solves": float(row["legalizer_solves"]),
                        "legalizer_ms": float(row["legalizer_ms"]),
                    },
                    "canonical_metrics": {
                        "legalized_grouping_violations": float(
                            row["legalized_grouping"]
                        ),
                        "legalized_mib_violations": float(
                            row["legalized_mib"]
                        ),
                        "legalized_boundary_violations": float(
                            row["legalized_boundary"]
                        ),
                        "legalized_hard_cost": float(
                            row["legalized_hard_cost"]
                        ),
                        "legalized_v_rel": float(row["legalized_v_rel"]),
                    },
                }
            )
    write_json_atomic(
        args.output,
        {
            "schema_version": "chip-curve-atom-rows-v1",
            "source_curve": str(args.curve),
            "source_summary": str(args.summary),
            "checkpoints": list(curve),
            "rows": rows,
        },
    )
    print(
        json.dumps(
            {
                "checkpoints": len(curve),
                "cases": len({row["case_id"] for row in rows}),
                "rows": len(rows),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

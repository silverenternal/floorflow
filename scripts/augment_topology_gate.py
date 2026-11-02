#!/usr/bin/env python3
"""Add GT-conditioned watertightness to an existing frozen prediction report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.cad_metric_suite import (
    balanced_cad_score,
    evaluate_hard_floors,
)
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-case-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = json.loads(args.per_case.read_text(encoding="utf-8"))["cases"]
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
    )
    by_name = {row["name"]: row for row in rows}
    conditional = []
    for index in range(args.offset, min(len(dataset), args.offset + len(rows))):
        target, _footprint, _floor_id, name = dataset[index]
        values = target.numpy()
        metrics = fpeval_raster_metrics(
            values,
            values,
            dataset.taxonomy,
            tolerance_fractions=(0.05,),
        )
        target_watertight = float(
            metrics["target_validity"]["watertight"]
        )
        row = by_name[name]
        row["target_watertight"] = target_watertight
        if target_watertight > 0.5:
            conditional.append(float(row["fpeval_watertight"]))
    conditional_watertight = float(np.mean(conditional)) if conditional else 0.0
    summary = report["cad_metrics"]["summary"]
    summary["conditional_watertight"] = conditional_watertight
    hard_floors = evaluate_hard_floors(summary)
    report["cad_metrics"]["hard_gate"] = {
        "passed": all(
            requirement["passed"]
            for requirement in hard_floors.values()
        ),
        "requirements": hard_floors,
    }
    report["cad_metrics"]["balanced_secondary"] = balanced_cad_score(summary)
    report["constraint_metrics"]["conditional_watertight"] = (
        conditional_watertight
    )
    report["topology_protocol"] = {
        "gt_watertight_cases": len(conditional),
        "cases": len(rows),
        "conditional_watertight": conditional_watertight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    args.per_case_output.parent.mkdir(parents=True, exist_ok=True)
    args.per_case_output.write_text(
        json.dumps({"cases": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "conditional_watertight": conditional_watertight,
                "hard_gate": report["cad_metrics"]["hard_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

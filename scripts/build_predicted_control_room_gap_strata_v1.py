#!/usr/bin/env python3
"""Stratify the current predicted-control room gap before model changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.core import extract_room_instances

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=ROOT / "outputs/reports/cad_furniture_dense_val64_eval_v1.json",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument(
        "--failure-taxonomy",
        type=Path,
        default=ROOT / "outputs/reports/room_failure_taxonomy_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/predicted_control_room_gap_strata_v1.json",
    )
    return parser.parse_args()


def _room_count(name: str, taxonomy: FloorplanTaxonomy, root: Path) -> int:
    category, case_id, floor = name.split("__", maxsplit=2)
    path = root / "val" / category / case_id / (
        floor.replace("-", "_") + ".npz"
    )
    with np.load(path) as payload:
        target = payload["target"].astype(np.int64)
    return len(extract_room_instances(target, taxonomy, min_area=8))


def _bin_name(count: int) -> str:
    if count <= 4:
        return "rooms_0_4"
    if count <= 7:
        return "rooms_5_7"
    return "rooms_8_plus"


def _aggregate(rows: list[dict]) -> dict[str, float]:
    keys = (
        "room_iou",
        "instance_f1",
        "instance_type_accuracy",
        "count_abs_error",
        "wall_f1",
        "opening_f1",
        "vector_conditional_watertight",
    )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in keys
    }


def main() -> None:
    args = parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    rows = []
    for row in evaluation["cases"]:
        count = _room_count(row["case"], taxonomy, args.cache_root)
        rows.append(
            {
                **row,
                "room_count": count,
                "room_count_bin": _bin_name(count),
            }
        )
    strata = {}
    for name in ("rooms_0_4", "rooms_5_7", "rooms_8_plus"):
        selected = [row for row in rows if row["room_count_bin"] == name]
        strata[name] = {
            "cases": len(selected),
            "metrics": _aggregate(selected) if selected else {},
        }
    taxonomy_report = json.loads(
        args.failure_taxonomy.read_text(encoding="utf-8")
    )
    report = {
        "schema_version": "predicted-control-room-gap-strata-v1",
        "evaluation_source": str(args.evaluation),
        "failure_taxonomy_source": str(args.failure_taxonomy),
        "cases": len(rows),
        "overall": _aggregate(rows),
        "room_count_strata": strata,
        "worst_cases": sorted(
            rows,
            key=lambda row: (
                row["room_iou"],
                row["instance_f1"],
            ),
        )[:10],
        "diagnosis": {
            "failure_categories": {
                name: values["fraction"]
                for name, values in taxonomy_report["categories"].items()
            },
            "primary_failure": "semantic_room_identity_confusion",
            "architecture_priority": (
                "instance-level room identity and count before boundary polish"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

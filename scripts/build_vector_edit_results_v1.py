#!/usr/bin/env python3
"""Build five-seed FEVP-2 vector results and operation strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import (
    load_metric_contract,
    tolerances_from_contract,
    validity_checks,
)
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.evaluation.typed_edit_metrics import normalized_region_mask
from floorflow.evaluation.vector_edit_protocol import (
    extract_vector_plan,
    vector_plan_preservation_metrics,
)
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan

ROOT = Path(__file__).resolve().parents[1]
EDITS = {
    "room_type": "fp_local_edit_room_type_test503_v1",
    "room_add": "fp_local_edit_room_add_test503_v1",
    "room_remove": "fp_local_edit_room_remove_test503_v1",
    "opening_delete": "fp_local_edit_opening_delete_test503_v1",
    "furniture_delete": "fp_local_edit_furniture_delete_test503_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 20, 21, 22, 23])
    parser.add_argument("--limit-per-operation", type=int, default=1000)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/fevp_metric_contract_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/vector_edit_results_v1.json",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _ci(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return [float(array[0]), float(array[0])]
    standard_error = array.std(ddof=1) / np.sqrt(array.size)
    critical = float(t.ppf(0.975, array.size - 1))
    return [
        float(array.mean() - critical * standard_error),
        float(array.mean() + critical * standard_error),
    ]


def _records(operation: str, limit: int) -> list[dict]:
    manifest = json.loads(
        (
            ROOT
            / "outputs/control_contracts"
            / EDITS[operation]
            / "test/manifest.json"
        ).read_text(encoding="utf-8")
    )
    return [row for row in manifest["rows"] if row["edited"]][: int(limit)]


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    tolerances = tolerances_from_contract(load_metric_contract(args.contract))
    records = {
        operation: _records(operation, args.limit_per_operation)
        for operation in EDITS
    }
    rows = []
    for seed in args.seeds:
        for operation, selected in records.items():
            for record in selected:
                safe_name = str(record["name"]).replace("/", "__")
                before = _load(
                    ROOT
                    / f"outputs/eval/fp_hard_metrics_full_test503_seed{seed}_v1/rasters"
                    / f"{safe_name}.npz"
                )
                after = _load(
                    ROOT
                    / f"outputs/eval/fp_control_gated_local_test503_seed{seed}_v1"
                    / operation
                    / f"{safe_name}.npz"
                )
                changed_cache = _load(
                    ROOT
                    / "outputs/cache/fp_edit_detection_v1"
                    / operation
                    / f"{safe_name}.npz"
                )
                footprint = np.asarray(before["footprint"], dtype=bool)
                region = normalized_region_mask(
                    footprint.shape,
                    record["edit_region_normalized"],
                ) & footprint
                target_after = np.asarray(before["target"]).copy()
                changed = changed_cache["changed"].astype(bool)
                target_after[changed] = changed_cache["after_visible"][changed]
                before_plan = extract_vector_plan(
                    before["prediction"],
                    before["room_labels"],
                    taxonomy,
                )
                after_plan = extract_vector_plan(
                    after["prediction"],
                    after["room_labels"],
                    taxonomy,
                )
                preservation = vector_plan_preservation_metrics(
                    before_plan,
                    after_plan,
                    region,
                    taxonomy,
                    changed_mask=(
                        np.asarray(before["prediction"])
                        != np.asarray(after["prediction"])
                    ),
                )
                validity = validity_checks(
                    raster_labels_to_wall_plan(
                        after["prediction"],
                        taxonomy,
                        room_labels=after["room_labels"],
                    ),
                    footprint,
                    taxonomy,
                    tolerances=tolerances,
                    room_labels=after["room_labels"],
                )
                graph = structural_legality_metrics(
                    after["prediction"],
                    target_after,
                    taxonomy,
                )
                rows.append(
                    {
                        "seed": seed,
                        "operation": operation,
                        "case_id": f"seed{seed}/{operation}/{record['name']}",
                        "outside_room_level_exact": preservation[
                            "outside_room_level_exact"
                        ],
                        "outside_wall_exact": preservation["outside_wall_exact"],
                        "outside_opening_exact": preservation[
                            "outside_opening_exact"
                        ],
                        "outside_edge_exact": preservation["outside_edge_exact"],
                        "locality_violation_area": preservation[
                            "locality_violation_area"
                        ],
                        "validity": validity["validity"],
                        "node_f1": graph["node_f1"],
                        "edge_overlap": graph["edge_overlap"],
                        "ged": graph["ged"],
                    }
                )
    metrics = (
        "outside_room_level_exact",
        "outside_wall_exact",
        "outside_opening_exact",
        "outside_edge_exact",
        "locality_violation_area",
        "validity",
        "node_f1",
        "edge_overlap",
        "ged",
    )
    by_operation = {}
    for operation in EDITS:
        operation_rows = [row for row in rows if row["operation"] == operation]
        by_operation[operation] = {}
        for metric in metrics:
            seed_values = [
                float(
                    np.mean(
                        [
                            row[metric]
                            for row in operation_rows
                            if row["seed"] == seed
                        ]
                    )
                )
                for seed in args.seeds
            ]
            by_operation[operation][metric] = {
                "seed_values": seed_values,
                "mean": float(np.mean(seed_values)),
                "ci95": _ci(seed_values),
                "seed_range": [
                    float(min(seed_values)),
                    float(max(seed_values)),
                ],
            }
    report = {
        "status": "complete",
        "seeds": args.seeds,
        "cases": len(rows),
        "cases_per_operation_per_seed": {
            operation: len(records[operation])
            for operation in EDITS
        },
        "ci_method": "student_t_interval_across_five_seeds",
        "by_operation": by_operation,
        "checks": {
            "outside_room_exact_one": all(
                by_operation[operation]["outside_room_level_exact"]["mean"] == 1.0
                for operation in EDITS
            ),
            "outside_wall_exact_one": all(
                by_operation[operation]["outside_wall_exact"]["mean"] == 1.0
                for operation in EDITS
            ),
            "outside_opening_exact_one": all(
                by_operation[operation]["outside_opening_exact"]["mean"] == 1.0
                for operation in EDITS
            ),
            "outside_edge_exact_one": all(
                by_operation[operation]["outside_edge_exact"]["mean"] == 1.0
                for operation in EDITS
            ),
            "locality_violation_zero": all(
                by_operation[operation]["locality_violation_area"]["mean"] == 0.0
                for operation in EDITS
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "checks": report["checks"]}, indent=2))
    raise SystemExit(0 if all(report["checks"].values()) else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

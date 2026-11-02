#!/usr/bin/env python3
"""Run the shared FEVP evaluators on 200 public CE2EPlan outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.ce2eplan_adapter import adapt_ce2eplan_row
from floorflow.evaluation.fevp_metric_contracts import (
    FEVPTolerances,
    load_metric_contract,
    tolerances_from_contract,
    validity_checks,
    watertight_diagnostics,
)
from floorflow.evaluation.housemind_structure import housemind_style_metrics

ROOT = Path(__file__).resolve().parents[1]
_TAXONOMY: FloorplanTaxonomy | None = None
_TOLERANCES: FEVPTolerances | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=ROOT / ".data/external/CE2EPlan",
    )
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
        default=ROOT / "outputs/reports/fevp_ce2eplan_portability_v3.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_ce2eplan_portability_v3.md",
    )
    return parser.parse_args()


def _initialize(taxonomy_path: str, contract_path: str) -> None:
    global _TAXONOMY, _TOLERANCES
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))
    _TOLERANCES = tolerances_from_contract(load_metric_contract(contract_path))


def _evaluate(task: dict) -> dict:
    if _TAXONOMY is None or _TOLERANCES is None:
        raise RuntimeError("worker globals were not initialized")
    prediction, target = task["prediction"], task["target"]
    labels, footprint, plan, rooms = adapt_ce2eplan_row(
        prediction,
        _TAXONOMY,
        prediction=True,
    )
    target_labels, target_footprint, _target_plan, target_rooms = (
        adapt_ce2eplan_row(
            target,
            _TAXONOMY,
            prediction=False,
        )
    )
    if not np.array_equal(footprint, target_footprint):
        raise ValueError(f"CE2EPlan boundary mismatch for {task['name']}")
    validity = validity_checks(
        plan,
        footprint,
        _TAXONOMY,
        tolerances=_TOLERANCES,
        room_labels=labels,
    )
    watertight = watertight_diagnostics(
        plan,
        taxonomy=_TAXONOMY,
        labels=labels,
        footprint=footprint,
        tolerances=_TOLERANCES,
    )
    structural = housemind_style_metrics(
        labels,
        target_labels,
        footprint,
        _TAXONOMY,
        ged_timeout_seconds=0.5,
    )
    return {
        "case_id": task["name"],
        "parsable": float(bool(rooms) and footprint.any()),
        "predicted_rooms": len(rooms),
        "target_rooms": len(target_rooms),
        "validity": validity,
        "watertight": watertight,
        "node_precision": structural["node_precision"],
        "node_recall": structural["node_recall"],
        "node_f1": structural["node_f1"],
        "edge_overlap": structural["edge_overlap"],
        "edge_precision": structural["edge_precision"],
        "edge_recall": structural["edge_recall"],
        "edge_f1": structural["edge_f1"],
        "ged": structural["ged"],
        "ged_exact": structural["exact"],
    }


def main() -> None:
    args = parse_args()
    target_path = args.external_root / "dataset/dataset_json/data_test.json"
    prediction_path = (
        args.external_root
        / "scripts/output/fp_ce2e_official_test12002_v1/b_auto.json"
    )
    target_rows = json.loads(target_path.read_text(encoding="utf-8"))
    prediction_rows = json.loads(prediction_path.read_text(encoding="utf-8"))
    target_by_name = {str(row["name"]): row for row in target_rows}
    indices = np.linspace(
        0,
        len(prediction_rows) - 1,
        num=min(args.cases, len(prediction_rows)),
        dtype=int,
    )
    sample_indices = [int(index) for index in indices]
    sample_indices_sha256 = hashlib.sha256(
        json.dumps(sample_indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    tasks = []
    for index in indices:
        prediction = prediction_rows[int(index)]
        name = str(prediction["name"])
        tasks.append(
            {
                "name": name,
                "prediction": prediction,
                "target": target_by_name[name],
            }
        )
    rows = []
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_initialize,
        initargs=(str(args.taxonomy), str(args.contract)),
    ) as executor:
        futures = [executor.submit(_evaluate, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"CE2EPlan FEVP audit {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: row["case_id"])
    metric_keys = (
        "parsable",
        "node_precision",
        "node_recall",
        "node_f1",
        "edge_precision",
        "edge_recall",
        "edge_f1",
        "edge_overlap",
        "ged",
        "ged_exact",
    )
    validity_keys = (
        "validity",
        "outer_boundary_closed",
        "room_polygon_closed",
        "no_self_intersection",
        "no_room_overlap",
        "semantic_complete",
    )
    watertight_keys = (
        "watertight_rate",
        "hole_count",
        "hole_area_ratio",
        "disconnected_component_count",
        "dangling_endpoint_rate",
    )
    report = {
        "status": "complete",
        "method": "CE2EPlan",
        "task": "RPLAN boundary-conditioned generation",
        "protocol": "FEVP v1.0-hardened portability adapter",
        "cases": len(rows),
        "sampling": "deterministic evenly spaced public test outputs",
        "sampling_method": "np.linspace_over_public_test_order",
        "sampling_seed": None,
        "sample_indices": sample_indices,
        "sample_indices_sha256": sample_indices_sha256,
        "outside_exact": "not applicable to generation-only outputs",
        "violation_rate": "not applicable to generation-only outputs",
        "metrics": {
            key: float(np.mean([row[key] for row in rows]))
            for key in metric_keys
        },
        "validity": {
            key: float(np.mean([row["validity"][key] for row in rows]))
            for key in validity_keys
        },
        "watertight": {
            key: float(np.mean([row["watertight"][key] for row in rows]))
            for key in watertight_keys
        },
        "validity_error_counts": dict(
            sorted(
                defaultdict(
                    int,
                    {
                        error: sum(
                            error in row["validity"]["validity_errors"]
                            for row in rows
                        )
                        for error in {
                            error
                            for row in rows
                            for error in row["validity"]["validity_errors"]
                        }
                    },
                ).items()
            )
        ),
        "checks": {
            "case_count_100_to_200": 100 <= len(rows) <= 200,
            "same_boundaries": True,
            "same_evaluator": True,
            "generation_outside_exact_not_claimed": True,
            "portable_validity_and_graph_metrics": (
                all(np.isfinite(report_mean) for report_mean in (
                    float(np.mean([row["validity"]["validity"] for row in rows])),
                    float(np.mean([row["node_f1"] for row in rows])),
                    float(np.mean([row["ged"] for row in rows])),
                ))
            ),
        },
        "claim": (
            "The FEVP evaluator is portable to a non-FloorFlow generation "
            "output: it decomposes CE2EPlan validity, watertightness, and graph "
            "quality on the same 200 public rows. This is not an editing or "
            "leaderboard comparison, and outside exact is not applicable."
        ),
        "sources": {
            "dataset": str(target_path.relative_to(ROOT)),
            "predictions": str(prediction_path.relative_to(ROOT)),
            "adapter": "src/floorflow/evaluation/ce2eplan_adapter.py",
        },
        "rows": rows,
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# CE2EPlan FEVP Portability Audit",
        "",
        "| Family | Metric | Value |",
        "|---|---|---:|",
    ]
    for key in metric_keys:
        table.append(f"| Graph / parsing | {key} | {report['metrics'][key]:.6f} |")
    for key in validity_keys:
        table.append(f"| Validity | {key} | {report['validity'][key]:.6f} |")
    for key in watertight_keys:
        table.append(
            f"| Watertightness | {key} | {report['watertight'][key]:.6f} |"
        )
    table.extend(
        [
            "",
            "Outside exact and violation rate: N/A because CE2EPlan is generation-only.",
            "",
            report["claim"],
            "",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

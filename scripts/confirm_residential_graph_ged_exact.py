"""Exact-recompute timed-out GED rows for the selected graph refinement."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.room_graph_edit import exact_room_graph_edit_distance
from floorflow.inference.room_graph_refinement import refine_room_graph

ROOT = Path(__file__).resolve().parents[1]
_TAXONOMY = None
_EVAL_ROOT = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "outputs/reports/residential_graph_refinement_final_val_v2.json",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT
        / "outputs/reports/residential_graph_refinement_final_val_rows_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/residential_graph_refinement_exact_v1.json",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=ROOT
        / "outputs/reports/residential_graph_refinement_exact_rows_v1.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--ged-timeout", type=float, default=60.0)
    return parser.parse_args()


def _initialize(eval_root: str, taxonomy_path: str) -> None:
    global _EVAL_ROOT, _TAXONOMY
    _EVAL_ROOT = Path(eval_root)
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))


def _recompute(row: dict) -> dict:
    if _TAXONOMY is None or _EVAL_ROOT is None:
        raise RuntimeError("GED worker is uninitialized")
    path = (
        _EVAL_ROOT
        / f"fp_local_edit_{row['operation']}_val191_v1"
        / "rasters"
        / f"{row['name']}.npz"
    )
    with np.load(path) as payload:
        prediction = payload["topology_prediction"]
        target = payload["target"]
        footprint = payload["footprint"].astype(bool)
    target_graph = refine_room_graph(
        target,
        _TAXONOMY,
        footprint=footprint,
        method="direct_tiled",
    )
    predicted_graph = refine_room_graph(
        prediction,
        _TAXONOMY,
        footprint=footprint,
        method="direct_tiled",
    )
    result = exact_room_graph_edit_distance(
        predicted_graph,
        target_graph,
        max_nodes=32,
        timeout_seconds=float(row["ged_timeout"]),
    )
    return {
        "operation": row["operation"],
        "name": row["name"],
        "ged": result["ged"],
        "ged_exact": result["ged_exact"],
        "fallback_reason": result["ged_fallback_reason"],
    }


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows_payload = json.loads(args.rows.read_text(encoding="utf-8"))
    rows = rows_payload["rows"]
    timed_out = [
        row for row in rows if not row["metrics"]["direct_tiled"]["ged_exact"]
    ]
    for row in timed_out:
        row["ged_timeout"] = args.ged_timeout
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_initialize,
        initargs=(
            str((ROOT / "outputs/eval").resolve()),
            str(
                (
                    ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
                ).resolve()
            ),
        ),
    ) as executor:
        updates = list(executor.map(_recompute, timed_out))
    by_key = {
        (row["operation"], row["name"]): row
        for row in rows
    }
    for update in updates:
        target = by_key[(update["operation"], update["name"])]
        target["metrics"]["direct_tiled"]["ged"] = update["ged"]
        target["metrics"]["direct_tiled"]["ged_exact"] = update["ged_exact"]
        target["metrics"]["direct_tiled"]["ged_fallback_reason"] = (
            update["fallback_reason"]
        )
    direct = [row["metrics"]["direct_tiled"] for row in rows]
    report["summary"]["direct_tiled"]["ged"] = float(
        np.mean([row["ged"] for row in direct])
    )
    report["summary"]["direct_tiled"]["ged_exact"] = float(
        np.mean([row["ged_exact"] for row in direct])
    )
    report["paired_deltas_vs_baseline"]["direct_tiled"]["ged"] = paired_delta(
        [
            {"value": row["metrics"]["direct_tiled"]["ged"]}
            for row in rows
        ],
        [
            {"value": row["metrics"]["wall_wall_primary"]["ged"]}
            for row in rows
        ],
        "value",
        iterations=args.bootstrap,
        seed=18001,
    )
    report["selected_ged_relative_reduction"] = (
        -report["paired_deltas_vs_baseline"]["direct_tiled"]["ged"][
            "mean_delta"
        ]
        / max(report["summary"]["wall_wall_primary"]["ged"], 1.0e-6)
    )
    report["checks"]["ged_exact_rate_at_least_0_95"] = (
        report["summary"]["direct_tiled"]["ged_exact"] >= 0.95
    )
    report["ged_exact_confirmation"] = {
        "recomputed_timeouts": len(updates),
        "remaining_timeouts": sum(
            not update["ged_exact"] for update in updates
        ),
        "timeout_seconds": args.ged_timeout,
    }
    report["status"] = "passed" if all(report["checks"].values()) else "failed"
    args.rows_output.write_text(
        json.dumps({"status": "complete", "rows": rows}) + "\n",
        encoding="utf-8",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "recomputed": len(updates),
                "ged_exact": report["summary"]["direct_tiled"]["ged_exact"],
                "status": report["status"],
            },
            indent=2,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

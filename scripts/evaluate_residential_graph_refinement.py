"""Evaluate deterministic room-graph refinement on validation-only edits."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.room_graph_edit import exact_room_graph_edit_distance
from floorflow.evaluation.room_graph_matching import (
    match_room_graphs,
)
from floorflow.inference.room_graph_refinement import (
    graph_refinement_methods,
    refine_room_graph,
)

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
PROTOCOL = "type_geometry_hungarian"
TARGET_METHOD = "direct_tiled"
_TAXONOMY = None
_EVAL_ROOT = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--ged-timeout", type=float, default=3.0)
    parser.add_argument(
        "--methods",
        default=(
            "wall_wall_primary,direct_tiled,tiled_purity_050,"
            "tiled_purity_050_contact_02,tiled_purity_050_contact_04,"
            "tiled_purity_050_contact_08,tiled_purity_050_wall_edges,"
            "tiled_purity_050_wall_union"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/residential_graph_refinement_v1.json",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=ROOT
        / "outputs/reports/residential_graph_refinement_rows_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/residential_graph_refinement_v1.md",
    )
    return parser.parse_args()


def _initialize(eval_root: str, taxonomy_path: str) -> None:
    global _EVAL_ROOT, _TAXONOMY
    _EVAL_ROOT = Path(eval_root)
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))


def _diagnose(task: dict) -> dict:
    if _TAXONOMY is None or _EVAL_ROOT is None:
        raise RuntimeError("graph refinement worker is uninitialized")
    path = (
        _EVAL_ROOT
        / f"fp_local_edit_{task['operation']}_val191_v1"
        / "rasters"
        / f"{task['name']}.npz"
    )
    with np.load(path) as payload:
        prediction = payload["topology_prediction"]
        target = payload["target"]
        footprint = payload["footprint"].astype(bool)
    target_graph = refine_room_graph(
        target,
        _TAXONOMY,
        footprint=footprint,
        method=TARGET_METHOD,
    )
    metrics = {}
    for method in task["methods"]:
        predicted_graph = refine_room_graph(
            prediction,
            _TAXONOMY,
            footprint=footprint,
            method=method,
        )
        matched = match_room_graphs(
            predicted_graph,
            target_graph,
            protocol=PROTOCOL,
        )
        ged = exact_room_graph_edit_distance(
            predicted_graph,
            target_graph,
            max_nodes=32,
            timeout_seconds=float(task["ged_timeout"]),
        )
        metrics[method] = {
            "node_precision": matched["node_precision"],
            "node_recall": matched["node_recall"],
            "node_f1": matched["node_f1"],
            "edge_overlap": matched["edge_overlap"],
            "ged": ged["ged"],
            "ged_exact": ged["ged_exact"],
            "predicted_nodes": len(predicted_graph.nodes),
            "target_nodes": len(target_graph.nodes),
            "predicted_edges": len(predicted_graph.edges),
            "target_edges": len(target_graph.edges),
        }
    return {
        "name": task["name"],
        "operation": task["operation"],
        "manifest_index": task["manifest_index"],
        "metrics": metrics,
    }


def _tasks(limit: int | None) -> list[dict]:
    tasks = []
    for operation in OPERATIONS:
        manifest = json.loads(
            (
                ROOT
                / f"outputs/control_contracts/fp_local_edit_{operation}_val191_v1"
                / "val/manifest.json"
            ).read_text(encoding="utf-8")
        )
        for index, row in enumerate(manifest["rows"]):
            if row["edited"]:
                tasks.append(
                    {
                        "operation": operation,
                        "name": row["name"].replace("/", "__"),
                        "manifest_index": index,
                        "methods": graph_refinement_methods(),
                        "ged_timeout": 3.0,
                    }
                )
    return tasks[:limit] if limit is not None else tasks


def _mean(rows: list[dict], method: str, key: str) -> float:
    return float(
        np.mean([row["metrics"][method][key] for row in rows])
    )


def _summary(rows: list[dict], methods: tuple[str, ...]) -> dict:
    keys = (
        "node_precision",
        "node_recall",
        "node_f1",
        "edge_overlap",
        "ged",
        "ged_exact",
        "predicted_nodes",
        "target_nodes",
        "predicted_edges",
        "target_edges",
    )
    return {
        method: {key: _mean(rows, method, key) for key in keys}
        for method in methods
    }


def _paired(rows: list[dict], method: str, key: str, args) -> dict:
    current = [{"value": row["metrics"][method][key]} for row in rows]
    baseline = [
        {"value": row["metrics"]["wall_wall_primary"][key]}
        for row in rows
    ]
    return paired_delta(
        current,
        baseline,
        "value",
        iterations=args.bootstrap,
        seed=18001,
    )


def _write_table(path: Path, summary: dict, deltas: dict, selected: str) -> None:
    lines = [
        "# Validation Room-Graph Refinement",
        "",
        f"Fixed evaluator: `{PROTOCOL}`; selected candidate: `{selected}`.",
        "",
        "| Readout | Node F1 | Edge overlap | Unit GED | Exact GED rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in summary.items():
        lines.append(
            f"| {method} | {values['node_f1']:.6f} | "
            f"{values['edge_overlap']:.6f} | {values['ged']:.6f} | "
            f"{values['ged_exact']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Paired delta vs baseline | Node F1 95% CI | Edge overlap 95% CI | GED 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, values in deltas.items():
        node = values["node_f1"]
        edge = values["edge_overlap"]
        ged = values["ged"]
        lines.append(
            f"| {method} | {node['mean_delta']:+.6f} "
            f"[{node['ci_low']:+.6f}, {node['ci_high']:+.6f}] | "
            f"{edge['mean_delta']:+.6f} "
            f"[{edge['ci_low']:+.6f}, {edge['ci_high']:+.6f}] | "
            f"{ged['mean_delta']:+.6f} "
            f"[{ged['ci_low']:+.6f}, {ged['ci_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            (
                "The refinement changes only the graph readout. Raster labels, "
                "controls, edit masks, and outside exactness are unchanged."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tasks = _tasks(args.limit)
    methods = tuple(method for method in args.methods.split(",") if method)
    allowed = set(graph_refinement_methods())
    if not methods or not set(methods) <= allowed:
        raise ValueError("unknown graph refinement method")
    for task in tasks:
        task["methods"] = methods
        task["ged_timeout"] = args.ged_timeout
    rows = []
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
        futures = [executor.submit(_diagnose, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 100 == 0 or completed == len(futures):
                print(f"refined {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (row["operation"], row["manifest_index"]))
    summary = _summary(rows, methods)
    candidates = [
        method for method in methods if method != "wall_wall_primary"
    ]
    selected = max(
        candidates,
        key=lambda method: (
            summary[method]["node_f1"]
            + summary[method]["edge_overlap"]
            - 0.01 * summary[method]["ged"]
        ),
    )
    deltas = {
        method: {
            key: _paired(rows, method, key, args)
            for key in ("node_f1", "edge_overlap", "ged")
        }
        for method in candidates
    }
    selected_delta = deltas[selected]
    ged_relative = -selected_delta["ged"]["mean_delta"] / max(
        summary["wall_wall_primary"]["ged"],
        1.0e-6,
    )
    checks = {
        "node_f1_gain_at_least_0_01": (
            selected_delta["node_f1"]["mean_delta"] >= 0.01
            and selected_delta["node_f1"]["ci_low"] > 0.0
        ),
        "edge_overlap_gain_at_least_0_01": (
            selected_delta["edge_overlap"]["mean_delta"] >= 0.01
            and selected_delta["edge_overlap"]["ci_low"] > 0.0
        ),
        "ged_relative_reduction_at_least_5pct": (
            ged_relative >= 0.05
            and selected_delta["ged"]["ci_high"] < 0.0
        ),
        "ged_exact_rate_at_least_0_95": (
            summary[selected]["ged_exact"] >= 0.95
        ),
        "no_raster_mutation": True,
    }
    report = {
        "schema_version": "residential-graph-refinement-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only",
        "split": "val191",
        "test_opened": False,
        "fixed_protocol": PROTOCOL,
        "fixed_target_method": TARGET_METHOD,
        "selected_candidate": selected,
        "cases": len(rows),
        "summary": summary,
        "paired_deltas_vs_baseline": deltas,
        "selected_ged_relative_reduction": ged_relative,
        "checks": checks,
        "operations": {
            operation: _summary(
                [row for row in rows if row["operation"] == operation],
                methods,
            )
            for operation in OPERATIONS
        },
    }
    args.rows_output.write_text(
        json.dumps({"status": "complete", "rows": rows}) + "\n",
        encoding="utf-8",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_table(args.table, summary, deltas, selected)
    print(json.dumps({"report": str(args.output), "selected": selected, "checks": checks}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

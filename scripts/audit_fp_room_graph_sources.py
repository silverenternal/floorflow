#!/usr/bin/env python3
"""Compare existing FloorFlow raster outputs under the canonical room graph."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.room_graph_matching import (
    INSTANCE_PRIMARY_PROTOCOL,
    build_direct_room_graph,
    match_room_graphs,
)

OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
SOURCES = ("topology_prediction", "prediction", "room_labels")
_TAXONOMY: FloorplanTaxonomy | None = None
_EVALUATION_ROOT: Path | None = None
_SEED = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("outputs/eval"),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("outputs/control_contracts"),
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path("outputs/reports/fp_room_graph_source_rows_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_room_graph_source_audit_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_room_graph_source_audit.md"),
    )
    return parser.parse_args()


def _initialize(taxonomy_path: str, evaluation_root: str, seed: int) -> None:
    global _TAXONOMY, _EVALUATION_ROOT, _SEED
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))
    _EVALUATION_ROOT = Path(evaluation_root)
    _SEED = int(seed)


def _diagnose(payload: dict) -> dict:
    if _TAXONOMY is None or _EVALUATION_ROOT is None:
        raise RuntimeError("worker was not initialized")
    path = (
        _EVALUATION_ROOT
        / f"fp_control_gated_local_test503_seed{_SEED}_v1"
        / payload["operation"]
        / f"{payload['name'].replace('/', '__')}.npz"
    )
    with np.load(path) as item:
        footprint = item["footprint"]
        target = build_direct_room_graph(
            item["target"],
            _TAXONOMY,
            footprint=footprint,
            tile_footprint=True,
        )
        graphs = {
            source: build_direct_room_graph(
                item[source],
                _TAXONOMY,
                footprint=footprint,
                tile_footprint=True,
            )
            for source in SOURCES
        }
    metrics = {}
    for source, graph in graphs.items():
        result = match_room_graphs(
            graph,
            target,
            protocol=INSTANCE_PRIMARY_PROTOCOL,
        )
        metrics[source] = {
            "node_precision": result["node_precision"],
            "node_recall": result["node_recall"],
            "node_f1": result["node_f1"],
            "edge_overlap": result["edge_overlap"],
            "matched_nodes": result["matched_nodes"],
            "predicted_nodes": len(graph.nodes),
            "target_nodes": len(target.nodes),
            "predicted_edges": len(graph.edges),
            "target_edges": len(target.edges),
        }
    return {
        "operation": payload["operation"],
        "name": payload["name"],
        "manifest_index": payload["manifest_index"],
        "metrics": metrics,
    }


def _mean(rows: list[dict], source: str, key: str) -> float:
    return float(np.mean([row["metrics"][source][key] for row in rows]))


def _summary(rows: list[dict]) -> dict:
    keys = (
        "node_precision",
        "node_recall",
        "node_f1",
        "edge_overlap",
        "matched_nodes",
        "predicted_nodes",
        "target_nodes",
        "predicted_edges",
        "target_edges",
    )
    return {
        source: {key: _mean(rows, source, key) for key in keys}
        for source in SOURCES
    }


def _paired(
    rows: list[dict],
    source: str,
    key: str,
    *,
    iterations: int,
) -> dict:
    current = [{"value": row["metrics"][source][key]} for row in rows]
    baseline = [
        {"value": row["metrics"]["topology_prediction"][key]} for row in rows
    ]
    return paired_delta(
        current,
        baseline,
        "value",
        iterations=iterations,
        seed=31,
    )


def _task_rows(args: argparse.Namespace) -> list[dict]:
    tasks = []
    for operation in OPERATIONS:
        manifest = json.loads(
            (
                args.control_root
                / f"fp_local_edit_{operation}_test503_v1"
                / "test"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        for index, record in enumerate(manifest["rows"]):
            if record["edited"]:
                tasks.append(
                    {
                        "operation": operation,
                        "name": record["name"],
                        "manifest_index": index,
                    }
                )
    return tasks[: args.limit] if args.limit is not None else tasks


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _table(summary: dict, deltas: dict) -> str:
    lines = [
        "# Existing Raster Outputs Under the Canonical Room Graph",
        "",
        "| Source | Node P | Node R | Node F1 | Edge overlap | Nodes | Edges |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, values in summary.items():
        lines.append(
            f"| {source} | {values['node_precision']:.6f} | "
            f"{values['node_recall']:.6f} | {values['node_f1']:.6f} | "
            f"{values['edge_overlap']:.6f} | "
            f"{values['predicted_nodes']:.3f} | "
            f"{values['predicted_edges']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Paired delta vs topology | Node F1 | 95% CI | Edge overlap | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for source, values in deltas.items():
        node = values["node_f1"]
        edge = values["edge_overlap"]
        lines.append(
            f"| {source} | {node['mean_delta']:+.6f} | "
            f"[{node['ci_low']:+.6f}, {node['ci_high']:+.6f}] | "
            f"{edge['mean_delta']:+.6f} | "
            f"[{edge['ci_low']:+.6f}, {edge['ci_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            (
                "All sources are existing frozen outputs. This audit selects no "
                "new model and uses oracle-control, non-deployable rows."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    tasks = _task_rows(args)
    rows = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_initialize,
        initargs=(
            str(args.taxonomy),
            str(args.evaluation_root),
            int(args.seed),
        ),
    ) as executor:
        futures = [executor.submit(_diagnose, task) for task in tasks]
        task_by_future = dict(zip(futures, tasks, strict=True))
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except (IndexError, KeyError, ValueError) as error:
                task = task_by_future[future]
                raise RuntimeError(
                    f"source audit failure for {task['operation']} {task['name']}"
                ) from error
            if completed % 250 == 0 or completed == len(futures):
                print(f"audited {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (row["operation"], row["manifest_index"]))
    summary = _summary(rows)
    deltas = {
        source: {
            "node_f1": _paired(rows, source, "node_f1", iterations=5000),
            "edge_overlap": _paired(
                rows,
                source,
                "edge_overlap",
                iterations=5000,
            ),
        }
        for source in SOURCES
        if source != "topology_prediction"
    }
    operations = {}
    for operation in OPERATIONS:
        selected = [row for row in rows if row["operation"] == operation]
        if selected:
            operations[operation] = _summary(selected)
    payload = {
        "status": "complete",
        "seed": args.seed,
        "cases": len(rows),
        "readout": "direct_tiled with label-plus-IoU matching",
        "sources": list(SOURCES),
        "summary": summary,
        "paired_deltas_vs_topology": deltas,
        "operations": operations,
        "boundary": (
            "The rows use target-derived oracle controls and are not deployable. "
            "Selecting the final model raster changes the structural appendix "
            "measurement only; it does not strengthen the edit-detection claim."
        ),
    }
    _write_json(args.rows_output, {
        "status": "complete",
        "seed": args.seed,
        "cases": len(rows),
        "rows": rows,
    })
    _write_json(args.output, payload)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_table(summary, deltas), encoding="utf-8")
    print(json.dumps({
        "cases": len(rows),
        "summary": summary,
        "deltas": deltas,
    }, indent=2))


if __name__ == "__main__":
    main()

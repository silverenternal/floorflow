#!/usr/bin/env python3
"""Compare canonical room-graph readouts and post-hoc identity alignment."""

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
    build_room_graph,
    filter_room_graph,
    match_room_graphs,
)

OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
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
        default=Path("outputs/reports/fp_room_identity_alignment_rows_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_room_identity_alignment_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_room_identity_alignment.md"),
    )
    return parser.parse_args()


def _initialize(
    taxonomy_path: str,
    evaluation_root: str,
    seed: int,
) -> None:
    global _TAXONOMY, _EVALUATION_ROOT, _SEED
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))
    _EVALUATION_ROOT = Path(evaluation_root)
    _SEED = int(seed)


def _graph_variants(
    prediction: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, tuple]:
    wall_prediction = build_room_graph(prediction, taxonomy)
    wall_target = build_room_graph(target, taxonomy)
    raw_prediction = build_direct_room_graph(
        prediction,
        taxonomy,
        adjacency_tolerance=2.0,
    )
    raw_target = build_direct_room_graph(
        target,
        taxonomy,
        adjacency_tolerance=2.0,
    )
    tiled_prediction = build_direct_room_graph(
        prediction,
        taxonomy,
        adjacency_tolerance=2.0,
        footprint=footprint,
        tile_footprint=True,
    )
    tiled_target = build_direct_room_graph(
        target,
        taxonomy,
        adjacency_tolerance=2.0,
        footprint=footprint,
        tile_footprint=True,
    )
    no_undefined_prediction = filter_room_graph(
        tiled_prediction,
        [
            node.index
            for node in tiled_prediction.nodes
            if node.label != "undefined"
        ],
    )
    no_undefined_target = filter_room_graph(
        tiled_target,
        [node.index for node in tiled_target.nodes if node.label != "undefined"],
    )
    pure_prediction = filter_room_graph(
        tiled_prediction,
        [node.index for node in tiled_prediction.nodes if node.label_purity >= 0.5],
    )
    pure_target = filter_room_graph(
        tiled_target,
        [node.index for node in tiled_target.nodes if node.label_purity >= 0.5],
    )
    return {
        "wall_wall_primary": (
            wall_prediction,
            wall_target,
            INSTANCE_PRIMARY_PROTOCOL,
        ),
        "direct_raw": (raw_prediction, raw_target, INSTANCE_PRIMARY_PROTOCOL),
        "direct_tiled": (
            tiled_prediction,
            tiled_target,
            INSTANCE_PRIMARY_PROTOCOL,
        ),
        "direct_tiled_label_iou": (
            tiled_prediction,
            tiled_target,
            "label_iou",
        ),
        "direct_tiled_no_undefined": (
            no_undefined_prediction,
            no_undefined_target,
            INSTANCE_PRIMARY_PROTOCOL,
        ),
        "direct_tiled_purity_ge_0_5": (
            pure_prediction,
            pure_target,
            INSTANCE_PRIMARY_PROTOCOL,
        ),
    }


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
        prediction = item["topology_prediction"]
        target = item["target"]
        footprint = item["footprint"]
    variants = _graph_variants(prediction, target, footprint, _TAXONOMY)
    metrics = {}
    for name, (predicted, expected, protocol) in variants.items():
        result = match_room_graphs(
            predicted,
            expected,
            protocol=protocol,
        )
        metrics[name] = {
            "protocol": protocol,
            "node_precision": result["node_precision"],
            "node_recall": result["node_recall"],
            "node_f1": result["node_f1"],
            "edge_overlap": result["edge_overlap"],
            "matched_nodes": result["matched_nodes"],
            "predicted_nodes": len(predicted.nodes),
            "target_nodes": len(expected.nodes),
            "predicted_edges": len(predicted.edges),
            "target_edges": len(expected.edges),
        }
    return {
        "operation": payload["operation"],
        "name": payload["name"],
        "manifest_index": payload["manifest_index"],
        "metrics": metrics,
    }


def _mean(rows: list[dict], variant: str, key: str) -> float:
    return float(np.mean([row["metrics"][variant][key] for row in rows]))


def _summary(rows: list[dict], variants: tuple[str, ...]) -> dict:
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
        variant: {key: _mean(rows, variant, key) for key in keys}
        for variant in variants
    }


def _operations(rows: list[dict], variants: tuple[str, ...]) -> dict:
    output = {}
    for operation in OPERATIONS:
        selected = [row for row in rows if row["operation"] == operation]
        if selected:
            output[operation] = _summary(selected, variants)
    return output


def _paired(
    rows: list[dict],
    candidate: str,
    key: str,
    *,
    iterations: int,
) -> dict:
    current = [
        {"value": row["metrics"][candidate][key]} for row in rows
    ]
    baseline = [
        {"value": row["metrics"]["wall_wall_primary"][key]} for row in rows
    ]
    result = paired_delta(
        current,
        baseline,
        "value",
        iterations=iterations,
        seed=23,
    )
    return result


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _table(summary: dict, deltas: dict) -> str:
    lines = [
        "# Room Identity Alignment Diagnostic",
        "",
        "| Readout | Node F1 | Edge overlap | Pred nodes | Target nodes |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, values in summary.items():
        lines.append(
            f"| {variant} | {values['node_f1']:.6f} | "
            f"{values['edge_overlap']:.6f} | "
            f"{values['predicted_nodes']:.3f} | {values['target_nodes']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Candidate delta vs primary | Node F1 | 95% CI | Edge overlap | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant, values in deltas.items():
        node = values["node_f1"]
        edge = values["edge_overlap"]
        lines.append(
            f"| {variant} | {node['mean_delta']:+.6f} | "
            f"[{node['ci_low']:+.6f}, {node['ci_high']:+.6f}] | "
            f"{edge['mean_delta']:+.6f} | "
            f"[{edge['ci_low']:+.6f}, {edge['ci_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            (
                "This is a graph-readout and post-hoc alignment audit. Pixels, "
                "labels, outside exactness, validity, and watertightness are "
                "unchanged."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _tasks(args: argparse.Namespace) -> list[dict]:
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


def main() -> None:
    args = parse_args()
    tasks = _tasks(args)
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
                    f"alignment failure for {task['operation']} {task['name']}"
                ) from error
            if completed % 100 == 0 or completed == len(futures):
                print(f"aligned {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (row["operation"], row["manifest_index"]))
    variants = tuple(rows[0]["metrics"]) if rows else ()
    summary = _summary(rows, variants)
    deltas = {
        variant: {
            "node_f1": _paired(rows, variant, "node_f1", iterations=5000),
            "edge_overlap": _paired(
                rows,
                variant,
                "edge_overlap",
                iterations=5000,
            ),
        }
        for variant in variants
        if variant != "wall_wall_primary"
    }
    payload = {
        "status": "complete",
        "seed": args.seed,
        "cases": len(rows),
        "primary_reference": "wall_wall_primary",
        "candidate": "direct_tiled",
        "definitions": {
            "wall_wall_primary": (
                "existing wall-face readout for prediction and target"
            ),
            "direct_raw": (
                "connected room-label polygons for prediction and target; "
                "2-pixel polygon-boundary adjacency"
            ),
            "direct_tiled": (
                "connected room-label polygons after nearest-room footprint "
                "tiling for prediction and target; 2-pixel adjacency"
            ),
            "direct_tiled_label_iou": (
                "direct_tiled graphs under same-label IoU sensitivity matching"
            ),
            "direct_tiled_no_undefined": (
                "direct_tiled after dropping undefined nodes from both graphs"
            ),
            "direct_tiled_purity_ge_0_5": (
                "direct_tiled after dropping nodes with label purity below 0.5"
            ),
        },
        "summary": summary,
        "paired_deltas_vs_primary": deltas,
        "operations": _operations(rows, variants),
        "decision": (
            "Use direct_tiled as the candidate canonical readout for structural "
            "metrics if its paired Node F1 gain clears 0.03 with CI excluding "
            "zero. Do not change pixel or edit-detection claims."
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

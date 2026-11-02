#!/usr/bin/env python3
"""Build matched-vs-stratified room-graph diagnostics for control-gated edits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_graph_matching import (
    INSTANCE_PRIMARY_PROTOCOL,
    MATCHING_PROTOCOLS,
    build_direct_room_graph,
    build_room_graph,
    protocol_matrix,
)
from floorflow.evaluation.room_graph_strata import (
    LOCATION_ORDER,
    EditRegion,
    ambiguity_summary,
    error_strata,
    ged_error_rows,
    quantile_summary,
    semantic_error_rows,
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
_READOUT = "wall"
_PREDICTION_KEY = "topology_prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ged-max-nodes", type=int, default=24)
    parser.add_argument("--ged-timeout", type=float, default=15.0)
    parser.add_argument(
        "--readout",
        choices=("wall", "direct_tiled"),
        default="wall",
    )
    parser.add_argument(
        "--prediction-key",
        choices=("topology_prediction", "prediction", "room_labels"),
        default="topology_prediction",
    )
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
        default=Path("outputs/reports/fp_room_graph_diagnostic_rows_v1.json"),
    )
    parser.add_argument(
        "--matches-output",
        type=Path,
        default=Path("outputs/reports/fp_room_graph_matching_protocol_v1.json"),
    )
    parser.add_argument(
        "--strata-output",
        type=Path,
        default=Path("outputs/reports/fp_structural_error_strata_v1.json"),
    )
    parser.add_argument(
        "--matching-table",
        type=Path,
        default=Path("paper/tables/fp_room_graph_matching_protocol.md"),
    )
    parser.add_argument(
        "--strata-table",
        type=Path,
        default=Path("paper/tables/fp_structural_error_strata.md"),
    )
    return parser.parse_args()


def _initialize(
    taxonomy_path: str,
    evaluation_root: str,
    seed: int,
    ged_max_nodes: int,
    ged_timeout: float,
    readout: str,
    prediction_key: str,
) -> None:
    global _TAXONOMY, _EVALUATION_ROOT, _SEED, _GED_MAX_NODES, _GED_TIMEOUT
    global _READOUT, _PREDICTION_KEY
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))
    _EVALUATION_ROOT = Path(evaluation_root)
    _SEED = int(seed)
    _GED_MAX_NODES = int(ged_max_nodes)
    _GED_TIMEOUT = float(ged_timeout)
    _READOUT = str(readout)
    _PREDICTION_KEY = str(prediction_key)


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
        prediction = item[_PREDICTION_KEY]
        target = item["target"]
        footprint = item["footprint"]
    if _READOUT == "direct_tiled":
        predicted = build_direct_room_graph(
            prediction,
            _TAXONOMY,
            footprint=footprint,
            tile_footprint=True,
        )
        expected = build_direct_room_graph(
            target,
            _TAXONOMY,
            footprint=footprint,
            tile_footprint=True,
        )
    else:
        predicted = build_room_graph(prediction, _TAXONOMY)
        expected = build_room_graph(target, _TAXONOMY)
    region = EditRegion.from_normalized(
        payload["edit_region_normalized"],
        width=target.shape[1],
        height=target.shape[0],
    )
    protocol = protocol_matrix(
        predicted,
        expected,
        ged_max_nodes=_GED_MAX_NODES,
        ged_timeout_seconds=_GED_TIMEOUT,
    )
    primary = protocol["matching"][INSTANCE_PRIMARY_PROTOCOL]
    semantic_errors = semantic_error_rows(predicted, expected, primary, region)
    ged_errors = ged_error_rows(
        predicted,
        expected,
        protocol["ged"]["mapping"],
        region,
        node_deletions=protocol["ged"]["_ged_node_deletions"],
        node_insertions=protocol["ged"]["_ged_node_insertions"],
        edge_deletions=protocol["ged"]["_ged_edge_deletions"],
        edge_insertions=protocol["ged"]["_ged_edge_insertions"],
    )
    matching = {
        name: {
            key: value
            for key, value in result.items()
            if key
            not in {
                "mapping",
                "node_false_positive_indices",
                "node_false_negative_indices",
                "edge_false_positive",
                "edge_false_negative",
            }
        }
        for name, result in protocol["matching"].items()
    }
    ged = {
        key: value
        for key, value in protocol["ged"].items()
        if key != "mapping" and not key.startswith("_ged_")
    }
    ambiguity = ambiguity_summary(predicted, expected)
    return {
        "operation": payload["operation"],
        "name": payload["name"],
        "manifest_index": payload["manifest_index"],
        "readout": _READOUT,
        "prediction_key": _PREDICTION_KEY,
        "predicted_node_count": len(predicted.nodes),
        "target_node_count": len(expected.nodes),
        "target_edge_count": len(expected.edges),
        "target_opening_count": expected.opening_count,
        "complexity_score": (
            len(expected.nodes) + len(expected.edges) + expected.opening_count
        ),
        "matching": matching,
        "ged": ged,
        "ambiguity": ambiguity,
        "errors": semantic_errors + ged_errors,
    }


def _mean(rows: list[dict], getter) -> float:
    return float(np.mean([getter(row) for row in rows])) if rows else 0.0


def _protocol_summary(rows: list[dict]) -> dict:
    summary = {}
    for protocol in MATCHING_PROTOCOLS:
        summary[protocol] = {
            key: _mean(
                rows,
                lambda row, protocol=protocol, key=key: row["matching"][protocol][key],
            )
            for key in (
                "node_precision",
                "node_recall",
                "node_f1",
                "edge_overlap",
                "matched_nodes",
                "node_substitutions",
            )
        }
    return summary


def _operation_summaries(rows: list[dict]) -> dict:
    result = {}
    for operation in OPERATIONS:
        selected = [row for row in rows if row["operation"] == operation]
        if selected:
            result[operation] = _protocol_summary(selected)
    return result


def _ged_summary(rows: list[dict]) -> dict:
    fallback_reasons = Counter(
        row["ged"]["ged_fallback_reason"]
        for row in rows
        if not row["ged"]["ged_exact"]
    )
    return {
        "mean": _mean(rows, lambda row: row["ged"]["ged"]),
        "median": float(np.median([row["ged"]["ged"] for row in rows])),
        "p90": float(np.quantile([row["ged"]["ged"] for row in rows], 0.9)),
        "exact_cases": sum(row["ged"]["ged_exact"] for row in rows),
        "fallback_cases": sum(not row["ged"]["ged_exact"] for row in rows),
        "exact_rate": _mean(rows, lambda row: float(row["ged"]["ged_exact"])),
        "fallback_reasons": dict(fallback_reasons),
        "components": {
            key: _mean(rows, lambda row, key=key: row["ged"][key])
            for key in (
                "node_deletion",
                "node_insertion",
                "node_substitution",
                "edge_deletion",
                "edge_insertion",
            )
        },
    }


def _error_totals(rows: list[dict]) -> dict:
    return error_strata(
        error for row in rows for error in row["errors"]
    )


def _ambiguity_totals(rows: list[dict]) -> dict:
    integer_keys = (
        "target_same_type_adjacent_edges",
        "target_low_purity_nodes",
        "target_ambiguous_nodes",
        "target_undefined_nodes",
        "predicted_undefined_nodes",
        "merged_predicted_nodes",
        "split_target_nodes",
    )
    return {
        "cases_with_same_type_adjacent_edges": sum(
            row["ambiguity"]["target_same_type_adjacent_edges"] > 0
            for row in rows
        ),
        "cases_with_low_purity_nodes": sum(
            row["ambiguity"]["target_low_purity_nodes"] > 0 for row in rows
        ),
        "cases_with_merges": sum(
            row["ambiguity"]["merged_predicted_nodes"] > 0 for row in rows
        ),
        "cases_with_splits": sum(
            row["ambiguity"]["split_target_nodes"] > 0 for row in rows
        ),
        **{
            key: int(sum(row["ambiguity"][key] for row in rows))
            for key in integer_keys
        },
        "mean_target_label_purity": _mean(
            rows,
            lambda row: row["ambiguity"]["mean_target_label_purity"],
        ),
    }


def _tail_share(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or array.sum() <= 0.0:
        return 0.0
    count = max(1, int(np.ceil(0.1 * array.size)))
    return float(np.sort(array)[-count:].sum() / array.sum())


def _strata_summary(rows: list[dict]) -> dict:
    metric_getters = {
        "primary_node_f1": lambda row: row["matching"][
            INSTANCE_PRIMARY_PROTOCOL
        ]["node_f1"],
        "primary_edge_overlap": lambda row: row["matching"][
            INSTANCE_PRIMARY_PROTOCOL
        ]["edge_overlap"],
        "exact_or_fallback_ged": lambda row: row["ged"]["ged"],
    }
    covariates = {
        "target_node_count": lambda row: row["target_node_count"],
        "target_edge_count": lambda row: row["target_edge_count"],
        "target_opening_count": lambda row: row["target_opening_count"],
        "complexity_score": lambda row: row["complexity_score"],
    }
    return {
        "error_counts": _error_totals(rows),
        "ambiguity": _ambiguity_totals(rows),
        "complexity": {
            metric: {
                covariate: quantile_summary(
                    [covariate_getter(row) for row in rows],
                    [metric_getter(row) for row in rows],
                )
                for covariate, covariate_getter in covariates.items()
            }
            for metric, metric_getter in metric_getters.items()
        },
        "tail_concentration": {
            "primary_node_loss_top_10pct_share": _tail_share(
                [1.0 - metric_getters["primary_node_f1"](row) for row in rows]
            ),
            "ged_top_10pct_share": _tail_share(
                [metric_getters["exact_or_fallback_ged"](row) for row in rows]
            ),
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _matching_table(summary: dict, ged: dict) -> str:
    lines = [
        "# Room-Graph Matching Protocol Sensitivity",
        "",
        "| Protocol | Node P | Node R | Node F1 | Edge overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for protocol in MATCHING_PROTOCOLS:
        values = summary[protocol]
        lines.append(
            f"| {protocol} | {values['node_precision']:.6f} | "
            f"{values['node_recall']:.6f} | {values['node_f1']:.6f} | "
            f"{values['edge_overlap']:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                f"Unit-cost GED mean: {ged['mean']:.6f}; exact on "
                f"{ged['exact_cases']} cases, fallback on "
                f"{ged['fallback_cases']}."
            ),
            "",
            (
                f"{INSTANCE_PRIMARY_PROTOCOL} is the frozen instance-graph "
                "protocol. Alternate rows are matching-sensitivity diagnostics "
                "and must not be selected as a headline."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _strata_table(strata: dict) -> str:
    lines = [
        "# Structural Error Strata",
        "",
        "| Error | Inside | Boundary 1 | Boundary 2 | Boundary 5 | Far outside |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind, values in strata["error_counts"].items():
        lines.append(
            f"| {kind} | "
            + " | ".join(str(values[location]) for location in LOCATION_ORDER)
            + " |"
        )
    lines.extend(
        [
            "",
            "| Metric / covariate | Spearman rho | p-value |",
            "|---|---:|---:|",
        ]
    )
    for metric, covariates in strata["complexity"].items():
        for covariate, result in covariates.items():
            rho = result["spearman"]
            p_value = result["p_value"]
            lines.append(
                f"| {metric} / {covariate} | "
                f"{'NA' if rho is None else f'{rho:.6f}'} | "
                f"{'NA' if p_value is None else f'{p_value:.6g}'} |"
            )
    lines.extend(
        [
            "",
            (
                "Node-loss top-10% share: "
                f"{strata['tail_concentration']['primary_node_loss_top_10pct_share']:.6f}."
            ),
            (
                "GED top-10% share: "
                f"{strata['tail_concentration']['ged_top_10pct_share']:.6f}."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _main_rows(
    args: argparse.Namespace,
) -> list[dict]:
    tasks = []
    for operation in OPERATIONS:
        manifest_path = (
            args.control_root
            / f"fp_local_edit_{operation}_test503_v1"
            / "test"
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for index, record in enumerate(manifest["rows"]):
            if not record["edited"]:
                continue
            tasks.append(
                {
                    "operation": operation,
                    "name": record["name"],
                    "manifest_index": index,
                    "edit_region_normalized": record["edit_region_normalized"],
                }
            )
    if args.limit is not None:
        tasks = tasks[: int(args.limit)]
    rows = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_initialize,
        initargs=(
            str(args.taxonomy),
            str(args.evaluation_root),
            int(args.seed),
            int(args.ged_max_nodes),
            float(args.ged_timeout),
            args.readout,
            args.prediction_key,
        ),
    ) as executor:
        futures = [executor.submit(_diagnose, task) for task in tasks]
        task_by_future = dict(zip(futures, tasks, strict=True))
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except IndexError as error:
                task = task_by_future[future]
                raise RuntimeError(
                    f"index failure for {task['operation']} "
                    f"{task['name']} at manifest row {task['manifest_index']}"
                ) from error
            if completed % 100 == 0 or completed == len(futures):
                print(f"diagnosed {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (row["operation"], row["manifest_index"]))
    return rows


def main() -> None:
    args = parse_args()
    rows = _main_rows(args)
    protocol_summary = _protocol_summary(rows)
    ged_summary = _ged_summary(rows)
    strata = _strata_summary(rows)
    deltas = {
        protocol: {
            "node_f1_delta": (
                protocol_summary[protocol]["node_f1"]
                - protocol_summary[INSTANCE_PRIMARY_PROTOCOL]["node_f1"]
            ),
            "edge_overlap_delta": (
                protocol_summary[protocol]["edge_overlap"]
                - protocol_summary[INSTANCE_PRIMARY_PROTOCOL]["edge_overlap"]
            ),
        }
        for protocol in MATCHING_PROTOCOLS
        if protocol != INSTANCE_PRIMARY_PROTOCOL
    }
    rows_payload = {
        "status": "complete",
        "seed": args.seed,
        "cases": len(rows),
        "primary_protocol": INSTANCE_PRIMARY_PROTOCOL,
        "readout": args.readout,
        "prediction_key": args.prediction_key,
        "rows": rows,
    }
    matching_payload = {
        "status": "complete",
        "seed": args.seed,
        "cases": len(rows),
        "primary_protocol": INSTANCE_PRIMARY_PROTOCOL,
        "readout": args.readout,
        "prediction_key": args.prediction_key,
        "protocol_definition": {
            "label_iou": "same room type; maximum IoU; accept at IoU >= 0.5",
            "geometry_hungarian": "type-agnostic maximum IoU; accept at IoU >= 0.5",
            "type_geometry_hungarian": (
                "maximum IoU plus a 0.1 same-type bonus; accept at IoU >= 0.5"
            ),
        },
        "summary": protocol_summary,
        "sensitivity_deltas_from_primary": deltas,
        "operations": _operation_summaries(rows),
        "ged": ged_summary,
        "decision": (
            f"Use {INSTANCE_PRIMARY_PROTOCOL} as the frozen instance-graph "
            "protocol; label-IoU remains a sensitivity row."
        ),
    }
    strata_payload = {
        "status": "complete",
        "seed": args.seed,
        "cases": len(rows),
        "readout": args.readout,
        "prediction_key": args.prediction_key,
        "location_definition": {
            "inside": "declared normalized edit rectangle",
            "boundary_1": "exclusive 1-pixel Chebyshev ring outside the rectangle",
            "boundary_2": "exclusive 2-pixel Chebyshev ring outside the rectangle",
            "boundary_5": "exclusive 5-pixel Chebyshev ring outside the rectangle",
            "far_outside": "more than 5 pixels outside the rectangle",
        },
        **strata,
    }
    _write_json(args.rows_output, rows_payload)
    _write_json(args.matches_output, matching_payload)
    _write_json(args.strata_output, strata_payload)
    args.matching_table.parent.mkdir(parents=True, exist_ok=True)
    args.matching_table.write_text(
        _matching_table(protocol_summary, ged_summary),
        encoding="utf-8",
    )
    args.strata_table.parent.mkdir(parents=True, exist_ok=True)
    args.strata_table.write_text(_strata_table(strata), encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": len(rows),
                "matching": protocol_summary,
                "ged": ged_summary,
                "tail": strata["tail_concentration"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

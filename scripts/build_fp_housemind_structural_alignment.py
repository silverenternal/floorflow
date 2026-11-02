#!/usr/bin/env python3
"""Align HouseMind-style structural metrics on CubiCasa edit rows."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.housemind_structure import housemind_style_metrics
from floorflow.evaluation.room_graph_matching import INSTANCE_PRIMARY_PROTOCOL

OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
COMPARISONS = (
    "before_target_vs_after_target",
    "before_prediction_vs_after_target",
    "after_prediction_vs_after_target",
)
_TAXONOMY: FloorplanTaxonomy | None = None
_BEFORE_ROOT: Path | None = None
_AFTER_ROOT: Path | None = None
_SEED = 19
_GED_TIMEOUT = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--ged-timeout", type=float, default=5.0)
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
        "--canonical-report",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path("outputs/reports/fp_housemind_structural_alignment_rows_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_housemind_structural_alignment_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_housemind_structural_alignment.md"),
    )
    return parser.parse_args()


def _initialize(
    taxonomy_path: str,
    before_root: str,
    after_root: str,
    seed: int,
    ged_timeout: float,
) -> None:
    global _TAXONOMY, _BEFORE_ROOT, _AFTER_ROOT, _SEED, _GED_TIMEOUT
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))
    _BEFORE_ROOT = Path(before_root)
    _AFTER_ROOT = Path(after_root)
    _SEED = int(seed)
    _GED_TIMEOUT = float(ged_timeout)


def _diagnose(payload: dict) -> dict:
    if _TAXONOMY is None or _BEFORE_ROOT is None or _AFTER_ROOT is None:
        raise RuntimeError("worker was not initialized")
    name = payload["name"].replace("/", "__")
    before_path = _BEFORE_ROOT / "rasters" / f"{name}.npz"
    after_path = (
        _AFTER_ROOT
        / f"fp_control_gated_local_test503_seed{_SEED}_v1"
        / payload["operation"]
        / f"{name}.npz"
    )
    with np.load(before_path) as before, np.load(after_path) as after:
        footprint = before["footprint"].astype(bool)
        metrics = {
            "before_target_vs_after_target": housemind_style_metrics(
                before["target"],
                after["target"],
                footprint,
                _TAXONOMY,
                ged_timeout_seconds=_GED_TIMEOUT,
            ),
            "before_prediction_vs_after_target": housemind_style_metrics(
                before["prediction"],
                after["target"],
                footprint,
                _TAXONOMY,
                ged_timeout_seconds=_GED_TIMEOUT,
            ),
            "after_prediction_vs_after_target": housemind_style_metrics(
                after["prediction"],
                after["target"],
                footprint,
                _TAXONOMY,
                ged_timeout_seconds=_GED_TIMEOUT,
            ),
        }
    return {
        "seed": _SEED,
        "operation": payload["operation"],
        "name": payload["name"],
        "manifest_index": payload["manifest_index"],
        "metrics": metrics,
    }


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
    return tasks


def _mean(rows: list[dict], comparison: str, key: str) -> float:
    return float(
        np.mean([row["metrics"][comparison][key] for row in rows])
    )


def _summary(rows: list[dict]) -> dict:
    keys = (
        "node_precision",
        "node_recall",
        "node_f1",
        "edge_overlap",
        "ged",
        "exact",
        "node_substitution",
        "node_deletion",
        "node_insertion",
        "edge_deletion",
        "edge_insertion",
    )
    return {
        comparison: {
            key: _mean(rows, comparison, key) for key in keys
        }
        for comparison in COMPARISONS
    }


def _operations(rows: list[dict]) -> dict:
    return {
        operation: _summary(
            [row for row in rows if row["operation"] == operation]
        )
        for operation in OPERATIONS
        if any(row["operation"] == operation for row in rows)
    }


def _canonical_summary(paths: list[Path] | None) -> dict:
    if not paths:
        return {}
    result = {}
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        matching = payload["summary"][INSTANCE_PRIMARY_PROTOCOL]
        result[str(seed)] = {
            "node_f1": matching["node_f1"],
            "edge_overlap": matching["edge_overlap"],
            "ged": payload["ged"]["mean"],
            "exact_ged_rate": payload["ged"]["exact_rate"],
            "source": str(path),
        }
    return result


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _table(per_seed: dict, canonical: dict) -> str:
    lines = [
        "# HouseMind-Style Structural Alignment on CubiCasa",
        "",
        (
            "HouseMind-E reference (RPLAN, natural language): Node F1 0.998, "
            "Edge overlap 0.934, GED 0.467. These values are not a "
            "same-dataset rank."
        ),
        "",
        "| Seed | Before GT NodeF1/Edge/GED | Before FloorFlow | After FloorFlow |",
        "|---:|---:|---:|---:|",
    ]
    for seed, values in per_seed.items():
        parts = []
        for comparison in COMPARISONS:
            item = values["summary"][comparison]
            parts.append(
                f"{item['node_f1']:.4f} / {item['edge_overlap']:.4f} / "
                f"{item['ged']:.4f}"
            )
        lines.append(f"| {seed} | " + " | ".join(parts) + " |")
    if canonical:
        lines.extend(
            [
                "",
                "| Seed | Canonical instance NodeF1 | Canonical Edge | Unit GED | Exact GED |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for seed, values in sorted(canonical.items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {seed} | {values['node_f1']:.6f} | "
                f"{values['edge_overlap']:.6f} | {values['ged']:.6f} | "
                f"{values['exact_ged_rate']:.6f} |"
            )
    lines.extend(
        [
            "",
            (
                "The HouseMind-style view collapses instances to unique room "
                "types and type-pair edges. The canonical view performs "
                "one-to-one room-instance IoU matching with explicit unit costs "
                "for edge edits."
            ),
            "",
            (
                "Both use oracle controls and CubiCasa test503; neither is a "
                "direct HouseMind-E RPLAN leaderboard reproduction."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    seeds = args.seed or [19, 20, 21, 22, 23]
    tasks = _tasks(args)
    per_seed = {}
    all_rows = []
    for seed in seeds:
        before_root = (
            args.evaluation_root
            / f"fp_hard_metrics_full_test503_seed{seed}_v1"
        )
        rows = []
        with ProcessPoolExecutor(
            max_workers=max(1, int(args.workers)),
            initializer=_initialize,
            initargs=(
                str(args.taxonomy),
                str(before_root),
                str(args.evaluation_root),
                int(seed),
                float(args.ged_timeout),
            ),
        ) as executor:
            futures = [executor.submit(_diagnose, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if completed % 500 == 0 or completed == len(futures):
                    print(f"seed {seed}: {completed}/{len(futures)}", flush=True)
        rows.sort(key=lambda row: (row["operation"], row["manifest_index"]))
        all_rows.extend(rows)
        per_seed[str(seed)] = {
            "cases": len(rows),
            "summary": _summary(rows),
            "operations": _operations(rows),
        }
    canonical = _canonical_summary(args.canonical_report)
    payload = {
        "status": "complete",
        "seeds": seeds,
        "cases_per_seed": len(tasks),
        "house_mind_reference": {
            "method": "HouseMind-E",
            "dataset": "HouseMind RPLAN shared test set",
            "cases": 2308,
            "input": "natural language",
            "delta_iou": 0.608,
            "delta_mse": 0.0019,
            "node_f1": 0.998,
            "edge_overlap": 0.934,
            "ged": 0.467,
            "source": "https://arxiv.org/abs/2603.11640 Table 3",
        },
        "house_mind_style_definition": (
            "Unique room-type nodes; unique room-type-pair adjacency edges; "
            "NetworkX GED with unit node substitution and default zero edge costs."
        ),
        "canonical_definition": (
            "Direct-tiled room instances; same-type IoU >= 0.5 matching; "
            "explicit unit costs for node and edge edits."
        ),
        "comparisons": {
            "before_target_vs_after_target": (
                "CubiCasa before-edit reference vs after-edit reference"
            ),
            "before_prediction_vs_after_target": (
                "FloorFlow before-edit prediction vs after-edit reference"
            ),
            "after_prediction_vs_after_target": (
                "FloorFlow control-gated after-edit prediction vs after-edit reference"
            ),
        },
        "per_seed": per_seed,
        "canonical_instance_view": canonical,
        "comparability": "not_directly_comparable",
        "boundaries": [
            "HouseMind-E uses RPLAN natural-language editing",
            "FloorFlow uses CubiCasa control-gated editing",
            "Structural results use oracle controls",
            "No HouseMind leaderboard rank is claimed"
        ],
    }
    _write_json(args.rows_output, {
        "status": "complete",
        "rows": all_rows,
    })
    _write_json(args.output, payload)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_table(per_seed, canonical), encoding="utf-8")
    print(json.dumps({
        "seeds": seeds,
        "cases_per_seed": len(tasks),
        "per_seed": per_seed,
    }, indent=2))


if __name__ == "__main__":
    main()

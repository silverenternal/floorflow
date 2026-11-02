#!/usr/bin/env python3
"""Audit final-output wall junctions against target junctions across seeds."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.structural_legality import wall_junction_summary
from floorflow.evaluation.wall_junction import wall_junction_metrics
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan

OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
SOURCES = ("prediction", "topology_prediction")
_TAXONOMY: FloorplanTaxonomy | None = None
_EVALUATION_ROOT: Path | None = None
_SEED = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=24)
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
        default=Path("outputs/reports/fp_wall_junction_rows_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_wall_junction_continuity_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_wall_junction_continuity.md"),
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
        target_walls = raster_labels_to_wall_plan(
            item["target"],
            _TAXONOMY,
        ).walls
        target = wall_junction_summary(target_walls)
        sources = {}
        for source in SOURCES:
            source_walls = raster_labels_to_wall_plan(
                item[source],
                _TAXONOMY,
            ).walls
            sources[source] = {
                **wall_junction_summary(source_walls),
                **wall_junction_metrics(source_walls, target_walls),
            }
    return {
        "seed": _SEED,
        "operation": payload["operation"],
        "name": payload["name"],
        "manifest_index": payload["manifest_index"],
        "sources": sources,
        "target": target,
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


def _mean(
    rows: list[dict],
    kind: str,
    source: str | None,
    key: str,
) -> float | None:
    values = [
        row["target"][key]
        if source is None
        else row["sources"][source][key]
        for row in rows
    ]
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else None


def _summary(rows: list[dict]) -> dict:
    keys = (
        "wall_continuity_rate",
        "junction_count",
        "open_endpoint_count",
        "degree_ge_2_rate",
        "degree_ge_3_rate",
        "junction_precision",
        "junction_recall",
        "junction_f1",
        "degree_exact_rate",
        "degree_mae",
        "mean_match_distance",
    )
    return {
        **{
            source: {
                key: _mean(rows, "source", source, key)
                for key in keys
                if key in rows[0]["sources"][source]
            }
            for source in SOURCES
        },
        "target": {
            key: _mean(rows, "target", None, key)
            for key in keys
            if key in rows[0]["target"]
        },
    }


def _paired(
    rows: list[dict],
    key: str,
    *,
    iterations: int,
) -> dict:
    current = [
        {"value": row["sources"]["prediction"][key]} for row in rows
    ]
    baseline = [
        {"value": row["sources"]["topology_prediction"][key]} for row in rows
    ]
    return paired_delta(
        current,
        baseline,
        "value",
        iterations=iterations,
        seed=37,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = args.seed or [19, 20, 21, 22, 23]
    tasks = _tasks(args)
    per_seed = {}
    all_rows = []
    for seed in seeds:
        rows = []
        with ProcessPoolExecutor(
            max_workers=max(1, int(args.workers)),
            initializer=_initialize,
            initargs=(
                str(args.taxonomy),
                str(args.evaluation_root),
                int(seed),
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
            "paired_prediction_minus_topology": {
                key: _paired(rows, key, iterations=3000)
                for key in (
                    "wall_continuity_rate",
                    "junction_f1",
                    "degree_exact_rate",
                    "open_endpoint_count",
                    "degree_ge_3_rate",
                )
            },
        }
    continuity = [
        item["summary"]["prediction"]["wall_continuity_rate"]
        for item in per_seed.values()
    ]
    target_continuity = [
        item["summary"]["target"]["wall_continuity_rate"]
        for item in per_seed.values()
    ]
    junction_f1 = [
        item["summary"]["prediction"]["junction_f1"]
        for item in per_seed.values()
    ]
    degree_exact = [
        item["summary"]["prediction"]["degree_exact_rate"]
        for item in per_seed.values()
    ]
    continuity_gap = [
        prediction - target
        for prediction, target in zip(continuity, target_continuity, strict=True)
    ]
    checks = {
        "junction_f1_at_least_0_65_all_seeds": min(junction_f1) >= 0.65,
        "degree_exact_at_least_0_75_all_seeds": min(degree_exact) >= 0.75,
        "continuity_gap_within_0_12_all_seeds": max(
            abs(value) for value in continuity_gap
        )
        <= 0.12,
        "positive_prediction_junction_f1_all_seeds": min(junction_f1) > 0.0,
    }
    payload = {
        "status": "complete",
        "seeds": seeds,
        "cases_per_seed": len(tasks),
        "definition": (
            "clustered 2-pixel endpoint junctions; Hungarian matching within "
            "6 pixels; degree strata L=2, T=3, cross>=4"
        ),
        "per_seed": per_seed,
        "checks": checks,
    }
    _write_json(args.rows_output, {
        "status": "complete",
        "rows": all_rows,
    })
    _write_json(args.output, payload)
    lines = [
        "# Wall-Junction Continuity Audit",
        "",
        (
            "| Seed | Junction F1 | Degree exact | Continuity pred | "
            "Continuity target | Gap |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, values in per_seed.items():
        summary = values["summary"]
        prediction = summary["prediction"]
        target = summary["target"]
        gap = (
            prediction["wall_continuity_rate"]
            - target["wall_continuity_rate"]
        )
        lines.append(
            f"| {seed} | {prediction['junction_f1']:.6f} | "
            f"{prediction['degree_exact_rate']:.6f} | "
            f"{prediction['wall_continuity_rate']:.6f} | "
            f"{target['wall_continuity_rate']:.6f} | {gap:+.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "Prediction range: "
                f"{min(continuity):.6f}-{max(continuity):.6f}."
            ),
            (
                "Target ceiling range: "
                f"{min(target_continuity):.6f}-{max(target_continuity):.6f}."
            ),
            f"Junction-F1 range: {min(junction_f1):.6f}-{max(junction_f1):.6f}.",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "per_seed": per_seed}, indent=2))


if __name__ == "__main__":
    main()

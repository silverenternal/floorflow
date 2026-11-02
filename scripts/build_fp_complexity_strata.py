#!/usr/bin/env python3
"""Build frozen complexity strata for strict-locality edit evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_graph_matching import INSTANCE_PRIMARY_PROTOCOL

OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
STRATUM_METRICS = (
    "delta_iou",
    "delta_mse",
    "outside_exact_ratio",
    "outside_room_exact_ratio",
    "validity",
    "conditional_watertight",
    "node_f1",
    "edge_overlap",
    "ged",
)
CONFIRMATORY_METRICS = ("delta_iou", "delta_mse", "node_f1", "ged")
_TAXONOMY: FloorplanTaxonomy | None = None
_EVALUATION_ROOT: Path | None = None
_SEED = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--bootstrap-iterations", type=int, default=3000)
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
        default=Path("outputs/reports/fp_complexity_strata_rows_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_complexity_strata_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_complexity_strata.md"),
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


def _region_mask(
    shape: tuple[int, int],
    normalized: list[float],
) -> np.ndarray:
    x, y, width, height = normalized
    rows, columns = shape
    left = max(0, min(columns, round(float(x) * columns)))
    top = max(0, min(rows, round(float(y) * rows)))
    right = max(left + 1, min(columns, round((x + width) * columns)))
    bottom = max(top + 1, min(rows, round((y + height) * rows)))
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _component_crossings(
    mask: np.ndarray,
    boundary: np.ndarray,
) -> int:
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0
    touching = np.unique(components[boundary & (components > 0)])
    return int(touching.size)


def _case_metrics(
    operation: str,
    record: dict,
    edit_row: dict,
    structural_row: dict,
    fpeval_row: dict,
    before_case: dict,
) -> dict:
    if _TAXONOMY is None or _EVALUATION_ROOT is None:
        raise RuntimeError("worker was not initialized")
    name = record["name"]
    safe_name = name.replace("/", "__")
    path = (
        _EVALUATION_ROOT
        / f"fp_control_gated_local_test503_seed{_SEED}_v1"
        / operation
        / f"{safe_name}.npz"
    )
    with np.load(path) as payload:
        target = payload["target"]
        footprint = payload["footprint"].astype(bool)
    region = _region_mask(target.shape, record["edit_region_normalized"])
    region = region & footprint
    boundary = binary_dilation(region, iterations=1) & footprint & ~region
    wall = target == _TAXONOMY.class_id("wall")
    opening = np.isin(target, np.asarray(_TAXONOMY.opening_ids))
    wall_crossings = _component_crossings(wall, boundary)
    opening_crossings = _component_crossings(opening, boundary)
    if wall_crossings and opening_crossings:
        interface = "both"
    elif wall_crossings:
        interface = "wall_only"
    elif opening_crossings:
        interface = "opening_only"
    else:
        interface = "neither"
    valid = bool(fpeval_row["prediction_validity"]["valid"])
    watertight = float(
        bool(fpeval_row["prediction_validity"]["watertight"])
        if float(before_case["target_watertight"]) > 0.5
        else True
    )
    return {
        "seed": _SEED,
        "operation": operation,
        "name": name,
        "manifest_index": record["_manifest_index"],
        "target_node_count": structural_row["target_node_count"],
        "edit_area_fraction": float(region.sum() / max(1, footprint.sum())),
        "wall_crossing_count": wall_crossings,
        "opening_crossing_count": opening_crossings,
        "interface_complexity": interface,
        "delta_iou": edit_row["local_delta_iou"],
        "delta_mse": edit_row["local_delta_mse"],
        "outside_exact_ratio": edit_row["outside_exact_ratio"],
        "outside_room_exact_ratio": edit_row["outside_room_exact_ratio"],
        "validity": float(valid),
        "conditional_watertight": watertight,
        "node_f1": structural_row["matching"][INSTANCE_PRIMARY_PROTOCOL]["node_f1"],
        "edge_overlap": structural_row["matching"][INSTANCE_PRIMARY_PROTOCOL][
            "edge_overlap"
        ],
        "ged": structural_row["ged"]["ged"],
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
                        "record": {**record, "_manifest_index": index},
                    }
                )
    return tasks


def _load_seed_rows(
    args: argparse.Namespace,
    seed: int,
    tasks: list[dict],
) -> list[dict]:
    _initialize(str(args.taxonomy), str(args.evaluation_root), seed)
    edit_reports = {}
    for operation in OPERATIONS:
        report = json.loads(
            (
                Path("outputs/reports")
                / f"fp_edit_detection_test503_controlgated_seed{seed}_v1.json"
            ).read_text(encoding="utf-8")
        )
        edit_reports[operation] = {
            row["name"]: row for row in report["operations"][operation]["rows"]
        }
    structural_path = Path("outputs/reports") / (
        "fp_room_graph_diagnostic_rows_final_v1.json"
        if seed == 19
        else f"fp_room_graph_diagnostic_rows_final_seed{seed}_v1.json"
    )
    structural_report = json.loads(structural_path.read_text(encoding="utf-8"))
    structural = {
        (row["operation"], row["name"]): row for row in structural_report["rows"]
    }
    fpeval = {}
    per_case = {}
    for operation in OPERATIONS:
        report = json.loads(
            (
                args.evaluation_root
                / f"fp_control_gated_topology_fpeval_{operation}_seed{seed}_v1.json"
            ).read_text(encoding="utf-8")
        )
        fpeval[operation] = {row["case"]: row for row in report["rows"]}
        case_dir = (
            f"fp_local_edit_{operation}_test503_v1"
            if seed == 19
            else f"fp_local_edit_{operation}_test503_seed{seed}_v1"
        )
        per_case[operation] = json.loads(
            (args.evaluation_root / case_dir / "per_case.json").read_text(
                encoding="utf-8"
            )
        )
    rows = []
    for task in tasks:
        operation = task["operation"]
        record = task["record"]
        name = record["name"]
        safe_name = name.replace("/", "__")
        rows.append(
            _case_metrics(
                operation,
                record,
                edit_reports[operation][name],
                structural[(operation, name)],
                fpeval[operation][safe_name],
                per_case[operation][record["_manifest_index"]],
            )
        )
    return rows


def _holm(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: row["p_value"])
    count = len(ordered)
    running = 0.0
    for rank, row in enumerate(ordered, start=1):
        value = min(1.0, (count - rank + 1) * row["p_value"])
        running = max(running, value)
        row["holm_q"] = running


def _cluster_arrays(
    rows: list[dict],
    metric: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["name"], []).append(float(row[metric]))
    keys = sorted(grouped)
    sums = np.asarray([sum(grouped[key]) for key in keys], dtype=np.float64)
    counts = np.asarray([len(grouped[key]) for key in keys], dtype=np.float64)
    return keys, sums, counts


def _bootstrap_means(
    sums: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return sums[indices].sum(axis=1) / counts[indices].sum(axis=1)


def _bootstrap_comparisons(
    rows: list[dict],
    strata: dict[str, dict[str, list[dict]]],
    *,
    iterations: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    cluster_keys = sorted({row["name"] for row in rows})
    cluster_index = {key: index for index, key in enumerate(cluster_keys)}
    sampled = rng.integers(
        0,
        len(cluster_keys),
        (iterations, len(cluster_keys)),
    )
    tests = []
    for family, bins in strata.items():
        for bin_name, bin_rows in bins.items():
            if not bin_rows:
                continue
            for metric in CONFIRMATORY_METRICS:
                overall_keys, overall_sums, overall_counts = _cluster_arrays(
                    rows, metric
                )
                bin_keys, bin_sums, bin_counts = _cluster_arrays(bin_rows, metric)
                overall_idx = np.asarray(
                    [cluster_index[key] for key in overall_keys],
                    dtype=np.int64,
                )
                bin_idx = np.asarray(
                    [cluster_index[key] for key in bin_keys],
                    dtype=np.int64,
                )
                overall_means = _bootstrap_means(
                    overall_sums,
                    overall_counts,
                    np.searchsorted(overall_idx, sampled),
                )
                # Build bin sums indexed by global cluster id for direct sampling.
                bin_sum_by_cluster = np.zeros(len(cluster_keys), dtype=np.float64)
                bin_count_by_cluster = np.zeros(len(cluster_keys), dtype=np.float64)
                bin_sum_by_cluster[bin_idx] = bin_sums
                bin_count_by_cluster[bin_idx] = bin_counts
                selected_sums = bin_sum_by_cluster[sampled]
                selected_counts = bin_count_by_cluster[sampled]
                denominator = selected_counts.sum(axis=1)
                valid = denominator > 0
                bin_means = np.full(iterations, np.nan, dtype=np.float64)
                bin_means[valid] = selected_sums[valid].sum(axis=1) / denominator[valid]
                differences = bin_means - overall_means
                finite = differences[np.isfinite(differences)]
                observed = float(
                    np.mean([row[metric] for row in bin_rows])
                    - np.mean([row[metric] for row in rows])
                )
                p_value = (
                    float(
                        2.0
                        * min(
                            np.mean(finite <= 0.0),
                            np.mean(finite >= 0.0),
                        )
                    )
                    if finite.size
                    else 1.0
                )
                tests.append(
                    {
                        "family": family,
                        "bin": bin_name,
                        "metric": metric,
                        "cases": len(bin_rows),
                        "mean": float(np.mean([row[metric] for row in bin_rows])),
                        "overall_mean": float(np.mean([row[metric] for row in rows])),
                        "mean_delta": observed,
                        "difference_ci_low": (
                            float(np.quantile(finite, 0.025)) if finite.size else None
                        ),
                        "difference_ci_high": (
                            float(np.quantile(finite, 0.975)) if finite.size else None
                        ),
                        "p_value": min(1.0, p_value),
                    }
                )
    _holm(tests)
    return tests


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _table(report: dict) -> str:
    lines = [
        "# Complexity Strata",
        "",
        "| Family | Bin | N | Delta-IoU | Delta-MSE | Outside exact | Validity | Watertight | NodeF1 | Edge | GED |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, bins in report["strata"].items():
        for bin_name, values in bins.items():
            metrics = values["metrics"]
            lines.append(
                f"| {family} | {bin_name} | {values['cases']} | "
                f"{metrics['delta_iou']:.6f} | {metrics['delta_mse']:.6f} | "
                f"{metrics['outside_exact_ratio']:.6f} | "
                f"{metrics['validity']:.6f} | "
                f"{metrics['conditional_watertight']:.6f} | "
                f"{metrics['node_f1']:.6f} | "
                f"{metrics['edge_overlap']:.6f} | {metrics['ged']:.6f} |"
            )
    lines.extend(
        [
            "",
            f"Edit-area cut points: {report['cut_points']['edit_area_fraction']}.",
            (
                "All rows are CubiCasa test503 oracle-control rows. Uncertainty "
                "uses floorplan-cluster bootstrap and Holm correction for "
                "confirmatory Delta-IoU, Delta-MSE, Node F1, and GED deviations."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    seeds = args.seed or [19, 20, 21, 22, 23]
    tasks = _tasks(args)
    rows = []
    for seed in seeds:
        seed_rows = _load_seed_rows(args, seed, tasks)
        rows.extend(seed_rows)
        print(f"seed {seed}: {len(seed_rows)} rows", flush=True)
    area_values = np.asarray([row["edit_area_fraction"] for row in rows])
    cut_points = np.quantile(area_values, [1.0 / 3.0, 2.0 / 3.0]).tolist()
    strata = {
        "room_count": {
            "0-4": [row for row in rows if row["target_node_count"] <= 4],
            "5-7": [
                row for row in rows if 5 <= row["target_node_count"] <= 7
            ],
            "8+": [row for row in rows if row["target_node_count"] >= 8],
        },
        "edit_area_fraction": {
            "small": [
                row for row in rows if row["edit_area_fraction"] < cut_points[0]
            ],
            "medium": [
                row
                for row in rows
                if cut_points[0] <= row["edit_area_fraction"] < cut_points[1]
            ],
            "large": [
                row for row in rows if row["edit_area_fraction"] >= cut_points[1]
            ],
        },
        "interface_complexity": {
            name: [row for row in rows if row["interface_complexity"] == name]
            for name in ("both", "wall_only", "opening_only", "neither")
        },
    }
    summary = {
        family: {
            name: {
                "cases": len(selected),
                "metrics": {
                    metric: float(np.mean([row[metric] for row in selected]))
                    if selected
                    else None
                    for metric in STRATUM_METRICS
                },
                "minimum": {
                    metric: float(np.min([row[metric] for row in selected]))
                    if selected
                    else None
                    for metric in (
                        "outside_exact_ratio",
                        "outside_room_exact_ratio",
                        "validity",
                        "conditional_watertight",
                    )
                },
            }
            for name, selected in bins.items()
        }
        for family, bins in strata.items()
    }
    tests = _bootstrap_comparisons(
        rows,
        strata,
        iterations=args.bootstrap_iterations,
        seed=41,
    )
    underpowered = [
        {"family": family, "bin": name, "cases": len(selected)}
        for family, bins in strata.items()
        for name, selected in bins.items()
        if len(selected) < 50
    ]
    report = {
        "status": "complete",
        "seeds": seeds,
        "cases": len(rows),
        "cut_points": {"edit_area_fraction": cut_points},
        "strata": summary,
        "confirmatory_tests": tests,
        "underpowered": underpowered,
        "checks": {
            "all_bins_supported": not underpowered,
            "outside_exact_one": all(
                values["minimum"]["outside_exact_ratio"] == 1.0
                for bins in summary.values()
                for values in bins.values()
            ),
            "outside_room_exact_one": all(
                values["minimum"]["outside_room_exact_ratio"] == 1.0
                for bins in summary.values()
                for values in bins.values()
            ),
            "validity_one": all(
                values["minimum"]["validity"] == 1.0
                for bins in summary.values()
                for values in bins.values()
            ),
            "watertight_at_least_0_95": all(
                values["metrics"]["conditional_watertight"] >= 0.95
                for bins in summary.values()
                for values in bins.values()
            ),
        },
        "boundary": (
            "Oracle-control CubiCasa strata, not a deployable or HouseMind "
            "leaderboard result."
        ),
    }
    _write_json(args.rows_output, {
        "status": "complete",
        "seeds": seeds,
        "rows": rows,
    })
    _write_json(args.output, report)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_table(report), encoding="utf-8")
    print(json.dumps({
        "cases": len(rows),
        "cut_points": cut_points,
        "checks": report["checks"],
        "underpowered": underpowered,
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build five-way validity and localized watertightness evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import (
    FEVPTolerances,
    load_metric_contract,
    tolerances_from_contract,
    validity_checks,
    watertight_diagnostics,
)
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
_TAXONOMY: FloorplanTaxonomy | None = None
_TOLERANCES: FEVPTolerances | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 20, 21, 22, 23])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/fevp_metric_contract_v2.json",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_validity_watertight_v2.json",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_validity_watertight_rows_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_validity_watertight_v2.md",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "paper/figures/fevp_watertight_failures_v2.pdf",
    )
    return parser.parse_args()


def _initialize(taxonomy_path: str, contract_path: str) -> None:
    global _TAXONOMY, _TOLERANCES
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))
    _TOLERANCES = tolerances_from_contract(load_metric_contract(contract_path))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _diagnose(task: dict) -> dict:
    if _TAXONOMY is None or _TOLERANCES is None:
        raise RuntimeError("worker globals were not initialized")
    seed = int(task["seed"])
    operation = task["operation"]
    index = int(task["index"])
    record = task["record"]
    name = str(record["name"])
    safe_name = name.replace("/", "__")
    raw_directory = (
        f"fp_local_edit_{operation}_test503_v1"
        if seed == 19
        else f"fp_local_edit_{operation}_test503_seed{seed}_v1"
    )
    raw = _load(ROOT / "outputs/eval" / raw_directory / "rasters" / f"{safe_name}.npz")
    gated = _load(
        ROOT
        / f"outputs/eval/fp_control_gated_local_test503_seed{seed}_v1"
        / operation
        / f"{safe_name}.npz"
    )
    per_case = task["per_case"]
    selected_watertight = bool(task["selected_watertight"])
    before_watertight = bool(float(per_case["fpeval_watertight"]) > 0.5)
    target_watertight = bool(float(per_case["target_watertight"]) > 0.5)
    plan = raster_labels_to_wall_plan(
        gated["topology_prediction"],
        _TAXONOMY,
        room_labels=gated["room_labels"],
    )
    validity = validity_checks(
        plan,
        gated["footprint"],
        _TAXONOMY,
        tolerances=_TOLERANCES,
        room_labels=gated["room_labels"],
    )
    diagnostics = watertight_diagnostics(
        plan,
        taxonomy=_TAXONOMY,
        labels=gated["topology_prediction"],
        footprint=gated["footprint"],
        fpeval_watertight=selected_watertight,
        tolerances=_TOLERANCES,
    )
    raw_plan = raster_labels_to_wall_plan(
        raw["prediction"],
        _TAXONOMY,
        room_labels=raw["room_labels"],
    )
    raw_diagnostics = watertight_diagnostics(
        raw_plan,
        taxonomy=_TAXONOMY,
        labels=raw["prediction"],
        footprint=raw["footprint"],
        fpeval_watertight=before_watertight,
        tolerances=_TOLERANCES,
    )
    return {
        "seed": seed,
        "operation": operation,
        "manifest_index": index,
        "case_id": f"seed{seed}/{operation}/{name}",
        "name": name,
        "target_watertight": target_watertight,
        "before_watertight": before_watertight,
        "after_watertight": selected_watertight,
        "repaired": bool(target_watertight and not before_watertight and selected_watertight),
        "regressed": bool(target_watertight and before_watertight and not selected_watertight),
        "validity": validity,
        "watertight": diagnostics,
        "raw_watertight": raw_diagnostics,
    }


def _tasks(seeds: list[int], limit: int | None) -> list[dict]:
    tasks = []
    for seed in seeds:
        for operation in OPERATIONS:
            manifest = json.loads(
                (
                    ROOT
                    / f"outputs/control_contracts/fp_local_edit_{operation}_test503_v1"
                    / "test/manifest.json"
                ).read_text(encoding="utf-8")
            )
            raw_directory = (
                f"fp_local_edit_{operation}_test503_v1"
                if seed == 19
                else f"fp_local_edit_{operation}_test503_seed{seed}_v1"
            )
            per_case = json.loads(
                (
                    ROOT
                    / "outputs/eval"
                    / raw_directory
                    / "per_case.json"
                ).read_text(encoding="utf-8")
            )
            topology = json.loads(
                (
                    ROOT
                    / "outputs/eval"
                    / f"fp_control_gated_topology_fpeval_{operation}_seed{seed}_v1.json"
                ).read_text(encoding="utf-8")
            )
            selected = {
                row["case"]: bool(row["prediction_validity"]["watertight"])
                for row in topology["rows"]
            }
            for index, record in enumerate(manifest["rows"]):
                if record["edited"]:
                    safe_name = str(record["name"]).replace("/", "__")
                    tasks.append(
                        {
                            "seed": seed,
                            "operation": operation,
                            "index": index,
                            "record": record,
                            "per_case": per_case[index],
                            "selected_watertight": selected[safe_name],
                        }
                    )
    return tasks[:limit] if limit is not None else tasks


def _summary(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["target_watertight"]]
    validity_keys = (
        "validity",
        "outer_boundary_closed",
        "room_polygon_closed",
        "no_self_intersection",
        "no_room_overlap",
        "semantic_complete",
    )
    failures_before = sum(
        row["target_watertight"] and not row["before_watertight"]
        for row in rows
    )
    return {
        "cases": len(rows),
        "watertight_eligible_cases": len(eligible),
        "before_watertight_rate": (
            float(np.mean([row["before_watertight"] for row in eligible]))
            if eligible
            else None
        ),
        "after_watertight_rate": (
            float(np.mean([row["after_watertight"] for row in eligible]))
            if eligible
            else None
        ),
        "repair_count": sum(row["repaired"] for row in rows),
        "regression_count": sum(row["regressed"] for row in rows),
        "repair_rate_of_failures": (
            sum(row["repaired"] for row in rows) / failures_before
            if failures_before
            else None
        ),
        "validity": {
            key: float(np.mean([row["validity"][key] for row in rows]))
            for key in validity_keys
        },
        "watertight": {
            key: float(np.mean([row["watertight"][key] for row in rows]))
            for key in (
                "watertight_rate",
                "hole_count",
                "hole_area_ratio",
                "disconnected_component_count",
                "dangling_endpoint_rate",
            )
        },
        "raw_watertight": {
            key: float(np.mean([row["raw_watertight"][key] for row in rows]))
            for key in (
                "watertight_rate",
                "hole_count",
                "hole_area_ratio",
                "disconnected_component_count",
                "dangling_endpoint_rate",
            )
        },
    }


def _group(rows: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        value: _summary(selected)
        for value, selected in sorted(grouped.items())
    }


def _render_failures(rows: list[dict], taxonomy: FloorplanTaxonomy, path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = sorted(
        [
            row
            for row in rows
            if row["target_watertight"]
            and not row["after_watertight"]
        ],
        key=lambda row: (
            -row["watertight"]["hole_count"],
            -row["watertight"]["dangling_endpoint_rate"],
        ),
    )[:8]
    if not selected:
        figure, axis = plt.subplots(figsize=(4.0, 2.0), constrained_layout=True)
        axis.text(
            0.5,
            0.5,
            "No watertight failures in this run",
            ha="center",
            va="center",
        )
        axis.axis("off")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        figure.savefig(path.with_suffix(".svg"))
        plt.close(figure)
        return
    colors = np.asarray(
        [taxonomy.color(index)[:3] for index in range(taxonomy.num_classes)],
        dtype=np.uint8,
    )
    figure, axes = plt.subplots(
        len(selected),
        2,
        figsize=(6.2, 1.8 * max(1, len(selected))),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, row in enumerate(selected):
        seed = row["seed"]
        operation = row["operation"]
        name = row["name"]
        gated = _load(
            ROOT
            / f"outputs/eval/fp_control_gated_local_test503_seed{seed}_v1"
            / operation
            / f"{name.replace('/', '__')}.npz"
        )
        axes[row_index, 0].imshow(colors[gated["topology_prediction"]])
        axes[row_index, 0].set_title(
            f"{operation} | holes={row['watertight']['hole_count']}",
            fontsize=7,
            loc="left",
        )
        axes[row_index, 0].axis("off")
        room_mask = np.isin(gated["topology_prediction"], taxonomy.room_ids)
        hole = (~room_mask) & gated["footprint"].astype(bool)
        overlay = np.zeros((*hole.shape, 4), dtype=np.float32)
        overlay[hole] = [0.9, 0.1, 0.1, 0.85]
        axes[row_index, 1].imshow(colors[gated["topology_prediction"]])
        axes[row_index, 1].imshow(overlay)
        axes[row_index, 1].set_title("Hole candidates", fontsize=7, loc="left")
        axes[row_index, 1].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    tasks = _tasks(args.seeds, args.limit)
    rows = []
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_initialize,
        initargs=(str(args.taxonomy), str(args.contract)),
    ) as executor:
        futures = [executor.submit(_diagnose, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 250 == 0 or completed == len(futures):
                print(f"validity audit {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (row["seed"], row["operation"], row["name"]))
    error_counts = Counter(
        error
        for row in rows
        for error in row["validity"]["validity_errors"]
    )
    report = {
        "status": "complete",
        "cases": len(rows),
        "summary": {
            "all": _summary(rows),
            "by_seed": _group(rows, "seed"),
            "by_operation": _group(rows, "operation"),
        },
        "validity_error_counts": dict(sorted(error_counts.items())),
        "watertight_issue_counts": dict(
            sorted(
                Counter(
                    issue
                    for row in rows
                    for issue in row["watertight"]["watertight_issue_codes"]
                ).items()
            )
        ),
        "checks": {
            "validity_is_five_check_conjunction": all(
                row["validity"]["validity"]
                == float(
                    all(
                        row["validity"][key] == 1.0
                        for key in (
                            "outer_boundary_closed",
                            "room_polygon_closed",
                            "no_self_intersection",
                            "no_room_overlap",
                            "semantic_complete",
                        )
                    )
                )
                for row in rows
            ),
            "repair_events_reported": any(row["repaired"] for row in rows),
            "watertight_failures_available": any(
                row["target_watertight"] and not row["after_watertight"]
                for row in rows
            ),
        },
    }
    report["status"] = (
        "complete" if all(report["checks"].values()) else "failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        json.dumps({"status": report["status"], "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]["all"]
    table = [
        "# FEVP Validity and Watertight Diagnostics",
        "",
        "| Check | Pass rate |",
        "|---|---:|",
    ]
    for key, value in summary["validity"].items():
        table.append(f"| {key} | {value:.6f} |")
    table.extend(
        [
            "",
            "| Watertight metric | Raw | Guarded |",
            "|---|---:|---:|",
        ]
    )
    for key in summary["watertight"]:
        table.append(
            f"| {key} | {summary['raw_watertight'][key]:.6f} | "
            f"{summary['watertight'][key]:.6f} |"
        )
    table.extend(
        [
            "",
            f"Repair count: {summary['repair_count']}",
            f"Regression count: {summary['regression_count']}",
            (
                "Repair rate among failures: "
                + (
                    f"{summary['repair_rate_of_failures']:.6f}"
                    if summary["repair_rate_of_failures"] is not None
                    else "N/A"
                )
            ),
            "",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    _render_failures(rows, FloorplanTaxonomy.read(args.taxonomy), args.figure)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(rows),
                **report["checks"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

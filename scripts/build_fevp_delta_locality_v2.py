#!/usr/bin/env python3
"""Build Delta distributions, strata, hard cases, and locality violation rates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import label as component_label

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.typed_edit_metrics import normalized_region_mask

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
_TAXONOMY: FloorplanTaxonomy | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 20, 21, 22, 23])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--typed-rows",
        type=Path,
        default=ROOT / "outputs/reports/fp_typed_edit_graph_ablation_rows_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_delta_locality_v2.json",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_delta_locality_rows_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_delta_locality_v2.md",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "paper/figures/fevp_delta_distributions_v2.pdf",
    )
    return parser.parse_args()


def _initialize(taxonomy_path: str) -> None:
    global _TAXONOMY
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _diagnose(task: dict) -> dict:
    if _TAXONOMY is None:
        raise RuntimeError("worker taxonomy was not initialized")
    seed = int(task["seed"])
    operation = task["operation"]
    report_row = task["report_row"]
    record = task["record"]
    name = str(report_row["name"])
    safe_name = name.replace("/", "__")
    before = _load(
        ROOT
        / f"outputs/eval/fp_hard_metrics_full_test503_seed{seed}_v1/rasters"
        / f"{safe_name}.npz"
    )
    with np.load(
        ROOT
        / "outputs/cache/fp_edit_detection_v1"
        / operation
        / f"{safe_name}.npz"
    ) as payload:
        changed = payload["changed"].astype(bool)
    region = normalized_region_mask(
        before["prediction"].shape,
        record["edit_region_normalized"],
    ) & before["footprint"].astype(bool)
    room_components, room_count = component_label(
        np.isin(before["target"], _TAXONOMY.room_ids)
    )
    changed_rooms = 0
    for component in range(1, room_count + 1):
        if np.any(changed & (room_components == component)):
            changed_rooms += 1
    edit_area_ratio = (
        float(region.sum() / max(1, before["footprint"].sum()))
    )
    outside_valid = int(report_row["outside_valid_pixels"])
    strict_outside_changes = int(report_row["outside_prediction_changes"])
    return {
        "seed": seed,
        "operation": operation,
        "case_id": name,
        "delta_iou": float(report_row["local_delta_iou"]),
        "delta_mse": float(report_row["local_delta_mse"]),
        "raw_delta_iou": float(report_row["raw_delta_iou"]),
        "raw_delta_mse": float(report_row["raw_delta_mse"]),
        "outside_exact": float(report_row["outside_exact_ratio"]),
        "strict_locality_violation_rate": (
            strict_outside_changes / outside_valid if outside_valid else 0.0
        ),
        "room_count": int(room_count),
        "changed_room_count": int(changed_rooms),
        "edit_area_ratio": edit_area_ratio,
    }


def _tasks(seeds: list[int], limit: int | None) -> list[dict]:
    tasks = []
    for seed in seeds:
        report = json.loads(
            (
                ROOT
                / f"outputs/reports/fp_edit_detection_test503_controlgated_seed{seed}_v1.json"
            ).read_text(encoding="utf-8")
        )
        for operation in OPERATIONS:
            manifest = json.loads(
                (
                    ROOT
                    / f"outputs/control_contracts/fp_local_edit_{operation}_test503_v1"
                    / "test/manifest.json"
                ).read_text(encoding="utf-8")
            )
            edited_records = []
            for record in manifest["rows"]:
                if record["edited"]:
                    edited_records.append(record)
            rows = report["operations"][operation]["rows"]
            if len(rows) != len(edited_records):
                raise ValueError(
                    f"row mismatch for {seed}/{operation}: "
                    f"{len(rows)} vs {len(edited_records)}"
                )
            for report_row, record in zip(rows, edited_records, strict=True):
                tasks.append(
                    {
                        "seed": seed,
                        "operation": operation,
                        "report_row": report_row,
                        "record": record,
                    }
                )
    return tasks[:limit] if limit is not None else tasks


def _room_bin(value: int) -> str:
    if value <= 4:
        return "rooms_0_4"
    if value <= 7:
        return "rooms_5_7"
    return "rooms_8_plus"


def _area_bin(value: float) -> str:
    if value < 0.10:
        return "edit_small_lt_0.10"
    if value < 0.25:
        return "edit_medium_0.10_0.25"
    return "edit_large_ge_0.25"


def _change_bin(value: int) -> str:
    if value <= 1:
        return "changed_rooms_1"
    if value <= 3:
        return "changed_rooms_2_3"
    return "changed_rooms_4_plus"


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _strata(rows: list[dict], key_fn, label: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return {
        value: {
            "cases": len(selected),
            "delta_iou": _distribution([row["delta_iou"] for row in selected]),
            "delta_mse": _distribution([row["delta_mse"] for row in selected]),
            "outside_exact_mean": float(
                np.mean([row["outside_exact"] for row in selected])
            ),
            "locality_violation_rate_mean": float(
                np.mean(
                    [row["strict_locality_violation_rate"] for row in selected]
                )
            ),
        }
        for value, selected in sorted(groups.items())
    } | {"definition": label}


def _render(rows: list[dict], typed_rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), constrained_layout=True)
    for axis, key, title in zip(
        axes,
        ("delta_iou", "delta_mse", "strict_locality_violation_rate"),
        ("Delta-IoU", "Delta-MSE", "Locality violation rate"),
        strict=True,
    ):
        values = [row[key] for row in rows]
        axis.hist(values, bins=80, color="#2f6fb6", alpha=0.85)
        axis.set_title(title, loc="left", fontsize=8, fontweight="bold")
        axis.grid(True, axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)

    by_arm = defaultdict(list)
    for row in typed_rows:
        for arm, values in row["arms"].items():
            by_arm[arm].append(values["locality_violation_rate"])
    figure, axis = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    names = list(by_arm)
    axis.boxplot(
        [by_arm[name] for name in names],
        tick_labels=names,
        showfliers=False,
    )
    axis.set_yscale("symlog", linthresh=1.0e-6)
    axis.set_title("Locality violation by routing arm", loc="left", fontsize=8)
    axis.grid(True, axis="y", alpha=0.25)
    figure.savefig(path.with_name(path.stem + "_arms" + path.suffix))
    figure.savefig(
        path.with_name(path.stem + "_arms" + path.suffix).with_suffix(".svg")
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    tasks = _tasks(args.seeds, args.limit)
    rows = []
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_initialize,
        initargs=(str(args.taxonomy),),
    ) as executor:
        futures = [executor.submit(_diagnose, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 500 == 0 or completed == len(futures):
                print(f"delta audit {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (row["seed"], row["operation"], row["case_id"]))
    for seed in sorted({row["seed"] for row in rows}):
        seed_rows = [row for row in rows if row["seed"] == seed]
        threshold = float(np.quantile([row["delta_iou"] for row in seed_rows], 0.10))
        for row in seed_rows:
            row["hard_tail"] = row["delta_iou"] <= threshold
            row["room_bin"] = _room_bin(row["room_count"])
            row["area_bin"] = _area_bin(row["edit_area_ratio"])
            row["change_bin"] = _change_bin(row["changed_room_count"])
    typed_source = json.loads(args.typed_rows.read_text(encoding="utf-8"))
    typed_rows = typed_source["rows"]
    arm_stats = {}
    for arm in sorted(typed_rows[0]["arms"]):
        arm_stats[arm] = {
            key: _distribution(
                [row["arms"][arm][key] for row in typed_rows]
            )
            for key in (
                "outside_pixel_exact",
                "locality_violation_rate",
                "delta_iou",
                "delta_mse",
                "node_f1",
                "edge_f1",
                "ged",
            )
        }
    hard = sorted(rows, key=lambda row: row["delta_iou"])[: max(1, len(rows) // 10)]
    report = {
        "status": "complete",
        "cases": len(rows),
        "hard_tail_cases": len(hard),
        "summary": {
            "delta_iou": _distribution([row["delta_iou"] for row in rows]),
            "delta_mse": _distribution([row["delta_mse"] for row in rows]),
            "strict_locality_violation_rate": _distribution(
                [row["strict_locality_violation_rate"] for row in rows]
            ),
            "by_operation": _strata(rows, lambda row: row["operation"], "operation"),
            "by_room_count": _strata(rows, lambda row: row["room_bin"], "room_count"),
            "by_edit_area": _strata(rows, lambda row: row["area_bin"], "edit_area"),
            "by_changed_rooms": _strata(
                rows,
                lambda row: row["change_bin"],
                "changed_room_count",
            ),
            "routing_arms": arm_stats,
        },
        "hard_tail": [
            {
                key: row[key]
                for key in (
                    "seed",
                    "operation",
                    "case_id",
                    "delta_iou",
                    "delta_mse",
                    "outside_exact",
                    "strict_locality_violation_rate",
                    "room_count",
                    "changed_room_count",
                    "edit_area_ratio",
                )
            }
            for row in hard[:50]
        ],
        "checks": {
            "all_requested_strata_present": all(
                len(report_groups) > 1
                for report_groups in (
                    _strata(rows, lambda row: row["operation"], "operation"),
                    _strata(rows, lambda row: row["room_bin"], "room_count"),
                    _strata(rows, lambda row: row["area_bin"], "edit_area"),
                    _strata(rows, lambda row: row["change_bin"], "changed_rooms"),
                )
            ),
            "strict_violation_rate_zero": all(
                row["strict_locality_violation_rate"] == 0.0 for row in rows
            ),
            "leaking_arm_has_positive_violation": (
                arm_stats["no_edit_gating"]["locality_violation_rate"]["mean"] > 0.0
            ),
        },
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        json.dumps({"status": report["status"], "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    table = [
        "# FEVP Delta and Locality Distributions",
        "",
        "| Metric | Mean | P10 | Median | P90 | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, title in (
        ("delta_iou", "Delta-IoU"),
        ("delta_mse", "Delta-MSE"),
        ("strict_locality_violation_rate", "Locality violation"),
    ):
        values = report["summary"][key]
        table.append(
            f"| {title} | {values['mean']:.6f} | {values['p10']:.6f} | "
            f"{values['median']:.6f} | {values['p90']:.6f} | "
            f"{values['min']:.6f} | {values['max']:.6f} |"
        )
    table.extend(
        [
            "",
            "| Arm | Violation mean | Violation P90 | Delta-IoU mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm, values in arm_stats.items():
        table.append(
            f"| {arm} | "
            f"{values['locality_violation_rate']['mean']:.9f} | "
            f"{values['locality_violation_rate']['p90']:.9f} | "
            f"{values['delta_iou']['mean']:.6f} |"
        )
    table.extend(["", "Operation, room-count, edit-area, and changed-room strata are in the JSON report.", ""])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    _render(rows, typed_rows, args.figure)
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build room-count and edit-area strata for FEVP-2 vector results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/reports/vector_edit_results_v1.json"
BENCHMARK = ROOT / "outputs/reports/vector_edit_benchmark_v1.json"
OUTPUT = ROOT / "outputs/reports/vector_edit_strata_v1.json"
TABLE = ROOT / "paper/tables/vector_edit_strata_v1.md"


def _ci(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return [float(array[0]), float(array[0])]
    error = array.std(ddof=1) / np.sqrt(array.size)
    critical = float(t.ppf(0.975, array.size - 1))
    return [float(array.mean() - critical * error), float(array.mean() + critical * error)]


def _room_bin(count: int) -> str:
    if count <= 4:
        return "rooms_0_4"
    if count <= 7:
        return "rooms_5_7"
    return "rooms_8_plus"


def _area_bin(ratio: float) -> str:
    if ratio <= 0.10:
        return "edit_area_small"
    if ratio <= 0.25:
        return "edit_area_medium"
    return "edit_area_large"


def _stats(rows: list[dict], metric: str, seeds: list[int]) -> dict:
    values = [
        float(
            np.mean(
                [
                    row[metric]
                    for row in rows
                    if row["seed"] == seed
                ]
            )
        )
        for seed in seeds
    ]
    return {
        "mean": float(np.mean(values)),
        "ci95": _ci(values),
        "seed_range": [float(min(values)), float(max(values))],
        "cases": len(rows),
    }


def main() -> None:
    result_report = json.loads(RESULTS.read_text(encoding="utf-8"))
    benchmark = {
        row["case_id"].split("/", 1)[1]: row
        for row in json.loads(BENCHMARK.read_text(encoding="utf-8"))["rows"]
    }
    seeds = [int(value) for value in result_report["seeds"]]
    rows = []
    for row in result_report["rows"]:
        key = row["case_id"].split("/", 1)[1]
        source = benchmark[key]
        edit_ratio = float(source["edit_region_area"]) / max(
            1.0,
            float(source["footprint_area"]),
        )
        enriched = dict(row)
        enriched["room_bin"] = _room_bin(
            int(source["denominators"]["before_rooms"])
        )
        enriched["area_bin"] = _area_bin(edit_ratio)
        rows.append(enriched)
    metrics = ("validity", "node_f1", "edge_overlap", "ged")
    strata = {}
    for dimension in ("room_bin", "area_bin"):
        strata[dimension] = {}
        for value in sorted({row[dimension] for row in rows}):
            selected = [row for row in rows if row[dimension] == value]
            strata[dimension][value] = {
                metric: _stats(selected, metric, seeds)
                for metric in metrics
            }
    report = {
        "status": "complete",
        "seeds": seeds,
        "cases": len(rows),
        "ci_method": "student_t_interval_across_five_seeds",
        "strata": strata,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# FEVP-2 Vector Strata v1",
        "",
        "Five-seed means with Student-t intervals.",
        "",
        "| Stratum | Cases | Validity | Node F1 | Edge overlap | GED |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dimension, values in strata.items():
        for name, row in values.items():
            lines.append(
                f"| {name} | {row['validity']['cases']} | "
                f"{row['validity']['mean']:.4f} | "
                f"{row['node_f1']['mean']:.4f} | "
                f"{row['edge_overlap']['mean']:.4f} | "
                f"{row['ged']['mean']:.4f} |"
            )
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "cases": len(rows)}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate hardened FEVP headline metrics over five seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outside",
        type=Path,
        default=ROOT / "outputs/reports/fevp_outside_layers_v2.json",
    )
    parser.add_argument(
        "--validity",
        type=Path,
        default=ROOT / "outputs/reports/fevp_validity_watertight_v2.json",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "outputs/reports/fevp_graph_metrics_v2.json",
    )
    parser.add_argument(
        "--structural",
        type=Path,
        default=ROOT / "outputs/reports/fp_housemind_structural_alignment_v1.json",
    )
    parser.add_argument(
        "--main",
        type=Path,
        default=ROOT / "outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json",
    )
    parser.add_argument(
        "--delta-rows",
        type=Path,
        default=ROOT / "outputs/reports/fevp_delta_locality_rows_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_fiveseed_stability_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_fiveseed_stability_v2.md",
    )
    return parser.parse_args()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(values: list[float], *, higher_is_better: bool) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    half_width = float(t.ppf(0.975, len(array) - 1) * std / np.sqrt(len(array)))
    best = float(array.max() if higher_is_better else array.min())
    worst = float(array.min() if higher_is_better else array.max())
    return {
        "values": [float(value) for value in array],
        "seed_range": [float(array.min()), float(array.max())],
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "ci_method": "student_t_interval_across_five_seeds",
        "ci_degrees_of_freedom": len(array) - 1,
        "best": best,
        "worst": worst,
        "best_seed": None,
        "worst_seed": None,
        "higher_is_better": higher_is_better,
    }


def main() -> None:
    args = parse_args()
    outside = _read(args.outside)
    validity = _read(args.validity)
    graph = _read(args.graph)
    structural = _read(args.structural)
    main = _read(args.main)
    delta_rows = _read(args.delta_rows)["rows"]
    main_by_seed = {
        int(seed): _read(
            ROOT
            / f"outputs/reports/fp_edit_detection_test503_controlgated_seed{seed}_v1.json"
        )["aggregate"]
        for seed in main["seeds"]
    }
    metrics = {
        "delta_iou": {},
        "delta_mse": {},
        "outside_pixel_exact": {},
        "outside_room_level_exact": {},
        "outside_edge_exact": {},
        "locality_violation_rate": {},
        "validity": {},
        "conditional_watertightness": {},
        "node_precision": {},
        "node_recall": {},
        "node_f1": {},
        "edge_overlap": {},
        "ged": {},
        "typed_edge_f1": {},
        "typed_ged": {},
    }
    seeds = sorted(int(seed) for seed in outside["summary"]["by_seed"])
    for seed in seeds:
        main_row = main_by_seed[seed]
        outside_row = outside["summary"]["by_seed"][str(seed)]["strict"]
        valid_row = validity["summary"]["by_seed"][str(seed)]
        graph_row = graph["per_seed"][str(seed)]["summary"]["typed_edit_graph"]
        structural_row = structural["per_seed"][str(seed)]["summary"][
            "after_prediction_vs_after_target"
        ]
        delta_values = [
            row["strict_locality_violation_rate"]
            for row in delta_rows
            if int(row["seed"]) == seed
        ]
        metrics["delta_iou"][seed] = float(main_row["local_delta_iou"])
        metrics["delta_mse"][seed] = float(main_row["local_delta_mse"])
        metrics["outside_pixel_exact"][seed] = float(
            outside_row["outside_pixel_exact"]
        )
        metrics["outside_room_level_exact"][seed] = float(
            outside_row["outside_room_level_exact"]
        )
        metrics["outside_edge_exact"][seed] = float(
            outside_row["outside_edge_exact"]
        )
        metrics["locality_violation_rate"][seed] = float(
            np.mean(delta_values)
        )
        metrics["validity"][seed] = float(valid_row["validity"]["validity"])
        metrics["conditional_watertightness"][seed] = float(
            valid_row["after_watertight_rate"]
        )
        metrics["node_precision"][seed] = float(structural_row["node_precision"])
        metrics["node_recall"][seed] = float(structural_row["node_recall"])
        metrics["node_f1"][seed] = float(structural_row["node_f1"])
        metrics["edge_overlap"][seed] = float(structural_row["edge_overlap"])
        metrics["ged"][seed] = float(structural_row["ged"])
        metrics["typed_edge_f1"][seed] = float(graph_row["edge_f1"])
        metrics["typed_ged"][seed] = float(graph_row["ged"])
    higher = {
        "delta_iou": True,
        "delta_mse": False,
        "outside_pixel_exact": True,
        "outside_room_level_exact": True,
        "outside_edge_exact": True,
        "locality_violation_rate": False,
        "validity": True,
        "conditional_watertightness": True,
        "node_precision": True,
        "node_recall": True,
        "node_f1": True,
        "edge_overlap": True,
        "ged": False,
        "typed_edge_f1": True,
        "typed_ged": False,
    }
    summary = {}
    for name, values in metrics.items():
        summary[name] = _summary(
            [values[seed] for seed in seeds],
            higher_is_better=higher[name],
        )
        ordered = sorted(values, key=lambda seed: values[seed], reverse=higher[name])
        summary[name]["best_seed"] = ordered[0]
        summary[name]["worst_seed"] = ordered[-1]
    report = {
        "status": "complete",
        "seeds": seeds,
        "metrics": summary,
        "checks": {
            "all_headline_metrics_present": all(
                len(values["values"]) == 5 for values in summary.values()
            ),
            "outside_room_exact_all_one": summary[
                "outside_room_level_exact"
            ]["worst"] == 1.0,
            "validity_all_at_least_0_97": summary["validity"]["worst"] >= 0.97,
            "watertight_all_at_least_0_95": summary[
                "conditional_watertightness"
            ]["worst"] >= 0.95,
        },
        "interpretation": (
            "Five seeds are deterministic stability samples. Confidence intervals "
            "describe seed variation and are not population-sampling inference."
        ),
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# FEVP Five-Seed Stability",
        "",
        "| Metric | Mean | SD | 95% CI | Best seed | Worst seed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary.items():
        table.append(
            f"| {name} | {values['mean']:.6f} | {values['std']:.6f} | "
            f"[{values['ci95_low']:.6f}, {values['ci95_high']:.6f}] | "
            f"{values['best_seed']} ({values['best']:.6f}) | "
            f"{values['worst_seed']} ({values['worst']:.6f}) |"
        )
    table.extend(
        [
            "",
            report["interpretation"],
            "",
            "CI method: Student-t interval over five seed means; not case-level bootstrap.",
            "",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

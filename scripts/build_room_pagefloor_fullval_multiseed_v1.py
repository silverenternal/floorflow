#!/usr/bin/env python3
"""Aggregate five-seed selected-readout results on full validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "room_macro_iou": False,
    "instance_room_f1": False,
    "instance_count_abs_error": True,
    "instance_type_accuracy": False,
    "room_support_iou": False,
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path, blend: float) -> list[dict]:
    return _read(path)["case_rows"][str(blend)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirmation-manifest",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/room_pagefloor_confirmation_manifest_v1.json"
        ),
    )
    parser.add_argument(
        "--fullval-manifest",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/"
            "room_pagefloor_confirmation_fullval_manifest_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/room_pagefloor_confirmation_val491_fiveseed_v1.json"
        ),
    )
    args = parser.parse_args()
    confirmation = _read(args.confirmation_manifest)
    fullval = _read(args.fullval_manifest)
    fullval_by_name = {row["name"]: Path(row["output"]) for row in fullval["rows"]}
    baseline_path = fullval_by_name["static_parent"]
    seed_reports = []
    for seed_row in confirmation["seeds"]:
        seed = int(seed_row["seed"])
        blend = float(seed_row["best_blend"])
        candidate_path = fullval_by_name[f"seed{seed}"]
        candidate_rows = _rows(candidate_path, blend)
        baseline_rows = _rows(baseline_path, blend)
        comparisons = {}
        for key, lower_is_better in METRICS.items():
            result = paired_delta(
                candidate_rows,
                baseline_rows,
                key,
                iterations=5000,
                seed=seed,
                lower_is_better=lower_is_better,
            )
            result["candidate_mean"] = mean(
                float(row[key]) for row in candidate_rows
            )
            comparisons[key] = result
        seed_reports.append(
            {
                "seed": seed,
                "blend": blend,
                "candidate_cases": str(candidate_path),
                "parent_cases": str(baseline_path),
                "comparisons": comparisons,
            }
        )
    aggregate = {}
    for key, lower_is_better in METRICS.items():
        values = [
            row["comparisons"][key]["candidate_mean"]
            for row in seed_reports
        ]
        deltas = [
            row["comparisons"][key]["mean_delta"]
            for row in seed_reports
        ]
        aggregate[key] = {
            "candidate_mean": mean(values),
            "candidate_sd": stdev(values),
            "candidate_min": min(values),
            "candidate_max": max(values),
            "delta_mean": mean(deltas),
            "delta_min": min(deltas),
            "delta_max": max(deltas),
            "improved_seeds": sum(
                (delta < 0.0) if lower_is_better else (delta > 0.0)
                for delta in deltas
            ),
            "ci_excludes_zero_seeds": sum(
                row["comparisons"][key]["ci_low"] > 0.0
                or row["comparisons"][key]["ci_high"] < 0.0
                for row in seed_reports
            ),
        }
    report = {
        "schema_version": "room-pagefloor-fullval-fiveseed-v1",
        "status": "complete",
        "cases": fullval["cases"],
        "seeds": len(seed_reports),
        "seed_reports": seed_reports,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

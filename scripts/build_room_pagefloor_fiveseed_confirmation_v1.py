#!/usr/bin/env python3
"""Aggregate full-train room confirmation evidence across five seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import wilcoxon

from floorflow.evaluation.paired_bootstrap import (
    clustered_paired_delta,
    paired_delta,
)
from floorflow.evaluation.statistical_effect_sizes import (
    rank_biserial_correlation,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "room_macro_iou": {"lower_is_better": False, "role": "primary"},
    "instance_room_f1": {"lower_is_better": False, "role": "primary"},
    "instance_count_abs_error": {
        "lower_is_better": True,
        "role": "primary",
    },
    "instance_type_accuracy": {
        "lower_is_better": False,
        "role": "non_regression",
    },
    "room_support_iou": {
        "lower_is_better": False,
        "role": "non_regression",
    },
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path, blend: float) -> list[dict]:
    rows = _read(path)["case_rows"][str(blend)]
    if "case" not in rows[0]:
        raise ValueError(f"case rows in {path} have no case key")
    return rows


def _beneficial(delta: float, *, lower_is_better: bool) -> bool:
    return delta < 0.0 if lower_is_better else delta > 0.0


def _wilcoxon(differences: np.ndarray) -> dict[str, float | int | str]:
    nonzero = differences[differences != 0.0]
    if not nonzero.size:
        return {
            "test": "wilcoxon_signed_rank",
            "nonzero_cases": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size": 0.0,
            "effect_size_method": "rank_biserial_correlation",
        }
    statistic, p_value = wilcoxon(nonzero, alternative="two-sided")
    return {
        "test": "wilcoxon_signed_rank",
        "nonzero_cases": int(nonzero.size),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": rank_biserial_correlation(nonzero),
        "effect_size_method": "rank_biserial_correlation",
    }


def _comparison(
    candidate: list[dict],
    baseline: list[dict],
    key: str,
    *,
    lower_is_better: bool,
    seed: int,
) -> dict:
    candidate_cases = [str(row["case"]) for row in candidate]
    baseline_cases = [str(row["case"]) for row in baseline]
    if candidate_cases != baseline_cases:
        raise ValueError("candidate and baseline case orders differ")
    result = paired_delta(
        candidate,
        baseline,
        key,
        iterations=5000,
        seed=seed,
        lower_is_better=lower_is_better,
    )
    differences = np.asarray(
        [
            float(current[key]) - float(reference[key])
            for current, reference in zip(candidate, baseline, strict=True)
        ]
    )
    result["wilcoxon"] = _wilcoxon(differences)
    result["ci_excludes_zero"] = bool(
        result["ci_low"] > 0.0 or result["ci_high"] < 0.0
    )
    return result


def _seed_summary(seed_reports: list[dict], key: str) -> dict:
    deltas = [row["comparisons"]["holdout"][key]["mean_delta"] for row in seed_reports]
    calibration = [
        row["comparisons"]["calibration"][key]["mean_delta"] for row in seed_reports
    ]
    lower_is_better = METRICS[key]["lower_is_better"]
    return {
        "calibration_mean": float(mean(calibration)),
        "calibration_min": float(min(calibration)),
        "calibration_max": float(max(calibration)),
        "holdout_mean": float(mean(deltas)),
        "holdout_sd": float(stdev(deltas)),
        "holdout_min": float(min(deltas)),
        "holdout_max": float(max(deltas)),
        "improved_seeds": sum(
            _beneficial(value, lower_is_better=lower_is_better)
            for value in deltas
        ),
        "ci_excludes_zero_seeds": sum(
            row["comparisons"]["holdout"][key]["ci_excludes_zero"]
            for row in seed_reports
        ),
    }


def _pooled(
    seed_reports: list[dict],
    key: str,
) -> dict:
    candidates = []
    baselines = []
    for row in seed_reports:
        candidate = _rows(
            Path(row["holdout_cases"]),
            float(row["best_blend"]),
        )
        baseline = _rows(
            Path(row["parent_holdout_cases"]),
            float(row["best_blend"]),
        )
        candidates.extend(candidate)
        baselines.extend(baseline)
    comparison = clustered_paired_delta(
        candidates,
        baselines,
        key,
        cluster_key="case",
        iterations=5000,
        seed=19,
        lower_is_better=METRICS[key]["lower_is_better"],
    )
    by_case: dict[str, list[float]] = {}
    for current, reference in zip(candidates, baselines, strict=True):
        by_case.setdefault(str(current["case"]), []).append(
            float(current[key]) - float(reference[key])
        )
    differences = np.asarray(
        [np.mean(values) for values in by_case.values()]
    )
    comparison["wilcoxon"] = _wilcoxon(differences)
    comparison["method"] = (
        "case-clustered paired bootstrap over seed repeats; "
        "Wilcoxon uses one seed-averaged difference per case"
    )
    return comparison


def _markdown(report: dict) -> str:
    lines = [
        "# Room Page-Floor Five-Seed Confirmation",
        "",
        "| Metric | Holdout delta mean | Range | Improved seeds | CI-excluding seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in METRICS:
        summary = report["aggregate"][key]
        lines.append(
            f"| {key} | {summary['holdout_mean']:+.6f} | "
            f"[{summary['holdout_min']:+.6f}, {summary['holdout_max']:+.6f}] | "
            f"{summary['improved_seeds']}/{report['seeds']} | "
            f"{summary['ci_excludes_zero_seeds']}/{report['seeds']} |"
        )
    lines.extend(
        [
            "",
            f"Gate passed: `{str(report['gate']['passed']).lower()}`.",
            "",
            "Case-clustered holdout bootstrap:",
            "",
            "| Metric | Mean delta | 95% CI | Wilcoxon p | Rank-biserial |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in METRICS:
        pooled = report["pooled_holdout"][key]
        lines.append(
            f"| {key} | {pooled['mean_delta']:+.6f} | "
            f"[{pooled['ci_low']:+.6f}, {pooled['ci_high']:+.6f}] | "
            f"{pooled['wilcoxon']['p_value']:.6g} | "
            f"{pooled['wilcoxon']['effect_size']:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/room_pagefloor_confirmation_manifest_v1.json"
        ),
    )
    parser.add_argument(
        "--parent-cal",
        type=Path,
        default=(
            ROOT / "outputs/reports/room_pagefloor_static_parent_cal384_cases_v1.json"
        ),
    )
    parser.add_argument(
        "--parent-holdout",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/room_pagefloor_static_parent_holdout107_cases_v1.json"
        ),
    )
    parser.add_argument(
        "--single-seed-report",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/room_pagefloor_confirmation_fiveseed_v1.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=(
            ROOT
            / "paper/tables/room_pagefloor_confirmation_fiveseed_v1.md"
        ),
    )
    args = parser.parse_args()
    manifest = _read(args.manifest)
    seed_reports = []
    for row in manifest["seeds"]:
        seed = int(row["seed"])
        blend = float(row["best_blend"])
        candidate_cal = _rows(Path(row["calibration_cases"]), blend)
        baseline_cal = _rows(args.parent_cal, blend)
        candidate_holdout = _rows(Path(row["holdout_cases"]), blend)
        baseline_holdout = _rows(args.parent_holdout, blend)
        comparisons = {"calibration": {}, "holdout": {}}
        for key, spec in METRICS.items():
            comparisons["calibration"][key] = _comparison(
                candidate_cal,
                baseline_cal,
                key,
                lower_is_better=spec["lower_is_better"],
                seed=seed,
            )
            comparisons["holdout"][key] = _comparison(
                candidate_holdout,
                baseline_holdout,
                key,
                lower_is_better=spec["lower_is_better"],
                seed=seed,
            )
        seed_reports.append(
            {
                **row,
                "parent_holdout_cases": str(args.parent_holdout),
                "comparisons": comparisons,
            }
        )
    aggregate = {
        key: _seed_summary(seed_reports, key)
        for key in METRICS
    }
    pooled = {
        key: _pooled(seed_reports, key)
        for key in METRICS
    }
    primary_pass = all(
        aggregate[key]["improved_seeds"] >= 4
        and aggregate[key]["ci_excludes_zero_seeds"] >= 4
        for key in METRICS
        if METRICS[key]["role"] == "primary"
    )
    non_regression = all(
        summary["holdout_mean"] >= 0.0
        for key, summary in aggregate.items()
        if METRICS[key]["role"] == "non_regression"
    )
    report = {
        "schema_version": "room-pagefloor-fiveseed-confirmation-v1",
        "status": "complete",
        "train_cases": manifest["train_cases"],
        "calibration": manifest["calibration"],
        "disjoint_holdout": manifest["disjoint_holdout"],
        "seeds": len(seed_reports),
        "seed_reports": seed_reports,
        "aggregate": aggregate,
        "pooled_holdout": pooled,
        "gate": {
            "passed": bool(primary_pass and non_regression),
            "primary_pass": bool(primary_pass),
            "non_regression": bool(non_regression),
            "rule": (
                "At least 4/5 seeds have the beneficial direction and a "
                "paired case bootstrap interval excluding zero for every "
                "primary metric; type and support do not regress."
            ),
        },
    }
    args.single_seed_report.parent.mkdir(parents=True, exist_ok=True)
    args.single_seed_report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

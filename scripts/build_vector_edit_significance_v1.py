#!/usr/bin/env python3
"""Paired FloorFlow versus Raster2Seq vector metrics across five seeds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.statistical_effect_sizes import (
    cliffs_delta,
    rank_biserial_correlation,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/reports/vector_edit_results_v1.json"
BASELINE = ROOT / "outputs/reports/vector_edit_baselines_v1.json"
OUTPUT = ROOT / "outputs/reports/vector_edit_significance_v1.json"
METRICS = (
    "outside_room_level_exact",
    "validity",
    "node_f1",
    "edge_overlap",
)


def _paired(values: list[float], baseline: list[float], seed: int) -> dict:
    differences = np.asarray(values) - np.asarray(baseline)
    bootstrap = paired_delta(
        [{"value": value} for value in values],
        [{"value": value} for value in baseline],
        "value",
        iterations=4000,
        seed=seed,
    )
    nonzero = differences[np.abs(differences) > 1.0e-12]
    if nonzero.size:
        test = wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        statistic = 0.0
        p_value = 1.0
    rank = float(rank_biserial_correlation(differences))
    delta = float(cliffs_delta(values, baseline))
    return {
        **bootstrap,
        "rank_biserial": rank,
        "cliffs_delta": delta,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_greater": p_value,
        "nonzero_pairs": int(nonzero.size),
        "higher_is_better": True,
    }


def main() -> None:
    result_rows = json.loads(RESULTS.read_text(encoding="utf-8"))["rows"]
    baseline_report = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_rows = [
        row
        for operation in baseline_report["operations"].values()
        for row in operation["rows"]
    ]
    baseline = {
        row["case_id"].split("/", 1)[1]: row
        for row in baseline_rows
    }
    by_seed = {}
    for seed in sorted({int(row["seed"]) for row in result_rows}):
        rows = [
            row
            for row in result_rows
            if int(row["seed"]) == seed
            and row["case_id"].split("/", 1)[1] in baseline
        ]
        comparisons = {}
        for index, metric in enumerate(METRICS):
            values = [float(row[metric]) for row in rows]
            external_key = metric
            baseline_values = [
                float(
                    baseline[row["case_id"].split("/", 1)[1]]["surgical"][
                        "preservation"
                    ][external_key]
                    if metric.startswith("outside_")
                    else baseline[row["case_id"].split("/", 1)[1]]["surgical"][
                        "topology"
                    ]["validity"]["validity"]
                    if metric == "validity"
                    else baseline[row["case_id"].split("/", 1)[1]]["surgical"][
                        "graph"
                    ][external_key]
                )
                for row in rows
            ]
            comparisons[metric] = _paired(
                values,
                baseline_values,
                seed=81000 + seed * 10 + index,
            )
        by_seed[str(seed)] = {
            "cases": len(rows),
            "comparisons": comparisons,
        }
    p_values = [
        row["comparisons"][metric]["wilcoxon_p_greater"]
        for row in by_seed.values()
        for metric in METRICS
    ]
    comparison_count = len(p_values)
    maximum_p = max(p_values)
    maximum_bonferroni_p = min(1.0, maximum_p * comparison_count)
    report = {
        "status": "complete",
        "scope": "FloorFlow versus fixed Raster2Seq surgical wrapper on matched cases",
        "ci_method": "paired_percentile_bootstrap",
        "multiplicity": {
            "method": "Bonferroni",
            "comparison_count": comparison_count,
            "maximum_raw_p": maximum_p,
            "maximum_adjusted_p": maximum_bonferroni_p,
            "familywise_alpha": 0.05,
        },
        "by_seed": by_seed,
        "checks": {
            "all_seed_ci_positive": all(
                row["comparisons"][metric]["ci_low"] > 0.0
                for row in by_seed.values()
                for metric in METRICS
            ),
            "all_seed_wilcoxon_p_lt_0_05": all(
                row["comparisons"][metric]["wilcoxon_p_greater"] < 0.05
                for row in by_seed.values()
                for metric in METRICS
            ),
            "all_seed_wilcoxon_bonferroni_lt_0_05": maximum_bonferroni_p
            < 0.05,
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "checks": report["checks"]}, indent=2))
    raise SystemExit(0 if all(report["checks"].values()) else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

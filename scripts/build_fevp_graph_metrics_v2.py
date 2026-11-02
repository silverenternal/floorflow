#!/usr/bin/env python3
"""Build decomposed graph metrics and fixed-evaluator paired GED tests."""

from __future__ import annotations

import argparse
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
ARMS = (
    "typed_edit_graph",
    "dense_cross_attention",
    "no_edit_gating",
    "no_structural_graph",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "outputs/reports/fp_typed_edit_graph_ablation_rows_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/fp_typed_edit_graph_ablation_v1.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/fevp_metric_contract_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_graph_metrics_v2.json",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_graph_rows_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_graph_metrics_v2.md",
    )
    return parser.parse_args()


def _paired(
    rows: list[dict],
    left: str,
    right: str,
    metric: str,
    *,
    seed: int,
) -> dict:
    left_values = [float(row["arms"][left][metric]) for row in rows]
    right_values = [float(row["arms"][right][metric]) for row in rows]
    differences = np.asarray(left_values) - np.asarray(right_values)
    bootstrap = paired_delta(
        [{"value": value} for value in left_values],
        [{"value": value} for value in right_values],
        "value",
        iterations=2000,
        seed=seed,
    )
    nonzero = differences[np.abs(differences) > 1.0e-12]
    if nonzero.size:
        test = wilcoxon(
            nonzero,
            alternative="less" if metric == "ged" else "greater",
            zero_method="wilcox",
        )
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        statistic = 0.0
        p_value = 1.0
    rank_biserial = float(rank_biserial_correlation(differences))
    delta = float(cliffs_delta(left_values, right_values))
    direction = 1.0 if metric != "ged" else -1.0
    return {
        **bootstrap,
        "median_delta": float(np.median(differences)),
        "rank_biserial": rank_biserial,
        "cliffs_delta": delta,
        "oriented_rank_biserial": direction * rank_biserial,
        "oriented_cliffs_delta": direction * delta,
        "oriented_effect_size_positive_favors_typed": True,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_less": p_value,
        "nonzero_pairs": int(nonzero.size),
        "lower_is_better": metric == "ged",
    }


def main() -> None:
    args = parse_args()
    source = json.loads(args.rows.read_text(encoding="utf-8"))
    report_source = json.loads(args.report.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    rows = source["rows"]
    by_seed = {}
    compact_rows = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        summary = {}
        for arm in ARMS:
            summary[arm] = {
                key: float(
                    np.mean([row["arms"][arm][key] for row in seed_rows])
                )
                for key in (
                    "node_precision",
                    "node_recall",
                    "node_f1",
                    "edge_precision",
                    "edge_recall",
                    "edge_f1",
                    "edge_overlap",
                    "ged",
                    "exact_ged",
                )
            }
        paired = {}
        for right_index, right in enumerate(ARMS[1:]):
            for metric_index, metric in enumerate(
                ("ged", "node_f1", "edge_f1")
            ):
                paired[f"typed_edit_graph_minus_{right}:{metric}"] = _paired(
                    seed_rows,
                    "typed_edit_graph",
                    right,
                    metric,
                    seed=73000 + seed * 10 + right_index * 3 + metric_index,
                )
        by_seed[str(seed)] = {
            "cases": len(seed_rows),
            "summary": summary,
            "paired": paired,
        }
        for row in seed_rows:
            compact_rows.append(
                {
                    "seed": seed,
                    "operation": row["operation"],
                    "case_id": row["name"],
                    "arms": {
                        arm: {
                            key: row["arms"][arm][key]
                            for key in (
                                "node_precision",
                                "node_recall",
                                "node_f1",
                                "edge_precision",
                                "edge_recall",
                                "edge_f1",
                                "edge_overlap",
                                "ged",
                                "exact_ged",
                            )
                        }
                        for arm in ARMS
                    },
                }
            )
    seed_means = {
        arm: {
            key: float(np.mean([by_seed[seed]["summary"][arm][key] for seed in by_seed]))
            for key in by_seed[next(iter(by_seed))]["summary"][arm]
        }
        for arm in ARMS
    }
    ged_p_values = [
        values["wilcoxon_p_less"]
        for seed in by_seed.values()
        for key, values in seed["paired"].items()
        if key.endswith(":ged")
    ]
    result = {
        "status": "complete",
        "scope": "five-seed matched typed-routing rows",
        "cases_per_seed": len(rows) // len(by_seed),
        "seeds": sorted(int(seed) for seed in by_seed),
        "matching_rules": {
            "nodes": contract["metrics"]["node_f1"]["matching"],
            "edges": contract["metrics"]["edge_overlap"]["matching"],
            "edge_contact_px": contract["tolerances"]["edge_contact_px"],
        },
        "ged_evaluator": contract["metrics"]["ged"],
        "summary": {
            "seed_mean": seed_means,
            "typed_ged_all_paired_tests_p_lt_0_05": all(
                p < 0.05 for p in ged_p_values
            ),
        },
        "per_seed": by_seed,
        "checks": {
            "all_arms_have_precision_recall_f1": all(
                all(
                    np.isfinite(row["arms"][arm][key])
                    for key in (
                        "node_precision",
                        "node_recall",
                        "node_f1",
                        "edge_precision",
                        "edge_recall",
                        "edge_f1",
                    )
                )
                for row in compact_rows
                for arm in ARMS
            ),
            "ged_exactness_reported": all(
                "exact_ged" in row["arms"][arm]
                for row in compact_rows
                for arm in ARMS
            ),
            "paired_tests_complete": all(
                len(seed["paired"]) == 9 for seed in by_seed.values()
            ),
        },
        "boundary": (
            "GED is interpreted only under one fixed evaluator and through paired "
            "differences; absolute values are not cross-paper leaderboard scores."
        ),
        "source_report_status": report_source["status"],
    }
    failures = [key for key, value in result["checks"].items() if not value]
    if failures:
        result["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        json.dumps({"status": result["status"], "rows": compact_rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    typed = seed_means["typed_edit_graph"]
    no_gate = seed_means["no_edit_gating"]
    dense = seed_means["dense_cross_attention"]
    no_graph = seed_means["no_structural_graph"]
    table = [
        "# FEVP Graph Metrics v2",
        "",
        "| Arm | Node P | Node R | Node F1 | Edge P | Edge R | Edge F1 | GED | Exact GED |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = seed_means[arm]
        table.append(
            f"| {arm} | {values['node_precision']:.6f} | "
            f"{values['node_recall']:.6f} | {values['node_f1']:.6f} | "
            f"{values['edge_precision']:.6f} | "
            f"{values['edge_recall']:.6f} | {values['edge_f1']:.6f} | "
            f"{values['ged']:.6f} | {values['exact_ged']:.6f} |"
        )
    table.extend(
        [
            "",
            (
                f"Typed GED {typed['ged']:.6f} versus no-gate "
                f"{no_gate['ged']:.6f}, dense {dense['ged']:.6f}, and graphless "
                f"{no_graph['ged']:.6f}."
            ),
            "GED comparisons use paired Wilcoxon tests and a fixed evaluator.",
            "",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["checks"]}, indent=2))
    raise SystemExit(0 if result["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

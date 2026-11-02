#!/usr/bin/env python3
"""Paired significance for inference-time vector residual refinement."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs/reports/vector_residual_inference_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/vector_residual_inference_significance_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/vector_residual_inference_v1.md",
    )
    return parser.parse_args()


def _paired(rows: list[dict], family: str, seed: int) -> dict:
    pairs = [
        (row[f"{family}_baseline"], row[f"{family}_refined"])
        for row in rows
        if row[f"{family}_baseline"] is not None
        and row[f"{family}_refined"] is not None
    ]
    baseline = [pair[0] for pair in pairs]
    refined = [pair[1] for pair in pairs]
    improvement = np.asarray(baseline) - np.asarray(refined)
    bootstrap = paired_delta(
        [{"value": value} for value in baseline],
        [{"value": value} for value in refined],
        "value",
        iterations=4000,
        seed=seed,
    )
    nonzero = improvement[np.abs(improvement) > 1.0e-12]
    test = wilcoxon(nonzero, alternative="greater", zero_method="wilcox") if nonzero.size else None
    return {
        "cases": len(pairs),
        "baseline_mae": float(np.mean(baseline)),
        "refined_mae": float(np.mean(refined)),
        "relative_reduction": float(
            (np.mean(baseline) - np.mean(refined)) / max(1.0e-12, np.mean(baseline))
        ),
        **bootstrap,
        "rank_biserial": float(rank_biserial_correlation(improvement)),
        "cliffs_delta": float(cliffs_delta(baseline, refined)),
        "wilcoxon_p_greater": float(test.pvalue) if test is not None else 1.0,
    }


def main() -> None:
    args = parse_args()
    report = json.loads(args.source.read_text(encoding="utf-8"))
    rows = report["rows"]
    operations = {}
    operation_names = sorted({row["operation"] for row in rows})
    for operation_index, operation in enumerate(operation_names):
        selected = [row for row in rows if row["operation"] == operation]
        operations[operation] = {
            family: _paired(
                selected,
                family,
                92000 + operation_index * 10 + family_index,
            )
            for family_index, family in enumerate(("wall", "opening"))
        }
    p_values = [
        values[family]["wilcoxon_p_greater"]
        for values in operations.values()
        for family in ("wall", "opening")
    ]
    comparison_count = len(p_values)
    maximum_p = max(p_values)
    maximum_bonferroni_p = min(1.0, maximum_p * comparison_count)
    checks = {
        "outside_exact_all_cases": all(row["outside_refined"] == 1.0 for row in rows),
        "outside_vector_layers_exact_all_cases": all(
            row["preservation"]["refined"][metric] == 1.0
            for row in rows
            for metric in (
                "outside_room_label_exact",
                "outside_room_polygon_exact",
                "outside_wall_exact",
                "outside_opening_exact",
                "outside_edge_exact",
            )
        ),
        "locality_violation_area_zero_all_cases": all(
            row["preservation"]["refined"]["locality_violation_area"] == 0.0
            for row in rows
        ),
        "wall_ci_positive_all_operations": all(
            row["wall"]["ci_low"] > 0.0 for row in operations.values()
        ),
        "opening_ci_positive_all_operations": all(
            row["opening"]["ci_low"] > 0.0 for row in operations.values()
        ),
        "wall_wilcoxon_all_operations": all(
            row["wall"]["wilcoxon_p_greater"] < 0.05
            for row in operations.values()
        ),
        "opening_wilcoxon_all_operations": all(
            row["opening"]["wilcoxon_p_greater"] < 0.05
            for row in operations.values()
        ),
        "all_wilcoxon_bonferroni_lt_0_05": maximum_bonferroni_p < 0.05,
    }
    result = {
        "status": "complete" if all(checks.values()) else "needs_iteration",
        "cases": len(rows),
        "room_refinement_enabled": report["room_refinement_enabled"],
        "multiplicity": {
            "method": "Bonferroni",
            "comparison_count": comparison_count,
            "maximum_raw_p": maximum_p,
            "maximum_adjusted_p": maximum_bonferroni_p,
            "familywise_alpha": 0.05,
        },
        "operations": operations,
        "checks": checks,
        "claim_boundary": (
            "Inference-time residual refinement is enabled only for wall "
            "endpoints and opening centers. Room polygons remain frozen."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Vector Residual Inference v1",
        "",
        "Room polygons remain frozen; only affected wall and opening proposals are refined.",
        "",
        "| Operation | Wall baseline | Wall refined | Reduction | Opening baseline | Opening refined | Reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for operation, values in operations.items():
        lines.append(
            f"| {operation} | {values['wall']['baseline_mae']:.6f} | "
            f"{values['wall']['refined_mae']:.6f} | "
            f"{values['wall']['relative_reduction']:.2%} | "
            f"{values['opening']['baseline_mae']:.6f} | "
            f"{values['opening']['refined_mae']:.6f} | "
            f"{values['opening']['relative_reduction']:.2%} |"
        )
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2))
    raise SystemExit(0 if result["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

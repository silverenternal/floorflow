#!/usr/bin/env python3
"""Aggregate vector residual inference over seeds 19-23."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t, wilcoxon

from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.statistical_effect_sizes import (
    cliffs_delta,
    rank_biserial_correlation,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = {
    19: "outputs/reports/vector_residual_inference_full_epoch2_500.json",
    20: "outputs/reports/vector_residual_inference_full_epoch2_seed20.json",
    21: "outputs/reports/vector_residual_inference_full_epoch2_seed21.json",
    22: "outputs/reports/vector_residual_inference_full_epoch2_seed22.json",
    23: "outputs/reports/vector_residual_inference_full_epoch2_seed23.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="Override a seed report; repeat for all five seeds.",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/train/vector_proposal_refiner_full_epoch2/best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/vector_residual_inference_multiseed_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/vector_residual_inference_multiseed_v1.md",
    )
    return parser.parse_args()


def _sources(overrides: list[str]) -> dict[int, str]:
    sources = dict(DEFAULT_SOURCES)
    for value in overrides:
        seed, separator, relative = value.partition("=")
        if not separator or not relative:
            raise ValueError(f"invalid --source value: {value!r}")
        sources[int(seed)] = relative
    return dict(sorted(sources.items()))


def _ci(values: list[float]) -> list[float]:
    array = np.asarray(values)
    error = array.std(ddof=1) / np.sqrt(array.size)
    critical = float(t.ppf(0.975, array.size - 1))
    return [float(array.mean() - critical * error), float(array.mean() + critical * error)]


def _paired(rows: list[dict], family: str, seed: int) -> dict:
    pairs = [
        (row[f"{family}_baseline"], row[f"{family}_refined"])
        for row in rows
        if row[f"{family}_baseline"] is not None
        and row[f"{family}_refined"] is not None
    ]
    baseline = [pair[0] for pair in pairs]
    refined = [pair[1] for pair in pairs]
    differences = np.asarray(baseline) - np.asarray(refined)
    nonzero = differences[np.abs(differences) > 1.0e-12]
    test = (
        wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
        if nonzero.size
        else None
    )
    bootstrap = paired_delta(
        [{"value": value} for value in baseline],
        [{"value": value} for value in refined],
        "value",
        iterations=4000,
        seed=seed,
    )
    return {
        "cases": len(pairs),
        "baseline_mae": float(np.mean(baseline)),
        "refined_mae": float(np.mean(refined)),
        "relative_reduction": float(
            (np.mean(baseline) - np.mean(refined)) / max(1.0e-12, np.mean(baseline))
        ),
        "ci_low": bootstrap["ci_low"],
        "ci_high": bootstrap["ci_high"],
        "ci_method": bootstrap["ci_method"],
        "rank_biserial": float(rank_biserial_correlation(differences)),
        "cliffs_delta": float(cliffs_delta(baseline, refined)),
        "wilcoxon_p_greater": float(test.pvalue) if test is not None else 1.0,
    }


def main() -> None:
    args = parse_args()
    sources = _sources(args.source)
    by_seed = {}
    refiner_hashes = set()
    for seed, relative in sources.items():
        report = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        refiner_hashes.add(report.get("refiner_sha256"))
        rows = report["rows"]
        operations = {}
        operation_names = sorted({row["operation"] for row in rows})
        for operation_index, operation in enumerate(operation_names):
            selected = [row for row in rows if row["operation"] == operation]
            operations[operation] = {
                family: _paired(
                    selected,
                    family,
                    seed=seed * 100 + operation_index * 2 + family_index,
                )
                for family_index, family in enumerate(("wall", "opening"))
            }
        by_seed[str(seed)] = {
            "cases": len(rows),
            "outside_exact_all_cases": all(
                row["outside_refined"] == 1.0 for row in rows
            ),
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
                row["preservation"]["refined"]["locality_violation_area"]
                == 0.0
                for row in rows
            ),
            "operations": operations,
        }
    seed_aggregate = {}
    for operation in sorted(
        next(iter(by_seed.values()))["operations"]
    ):
        seed_aggregate[operation] = {}
        for family in ("wall", "opening"):
            reductions = [
                by_seed[str(seed)]["operations"][operation][family][
                    "relative_reduction"
                ]
                for seed in sources
            ]
            seed_aggregate[operation][family] = {
                "seed_values": reductions,
                "mean_relative_reduction": float(np.mean(reductions)),
                "ci95": _ci(reductions),
                "seed_range": [float(min(reductions)), float(max(reductions))],
            }
    p_values = [
        row[family]["wilcoxon_p_greater"]
        for seed in by_seed.values()
        for row in seed["operations"].values()
        for family in ("wall", "opening")
    ]
    comparison_count = len(p_values)
    maximum_p = max(p_values)
    maximum_bonferroni_p = min(1.0, maximum_p * comparison_count)
    checks = {
        "outside_exact_all_seeds": all(
            row["outside_exact_all_cases"] for row in by_seed.values()
        ),
        "outside_vector_layers_exact_all_seeds": all(
            row["outside_vector_layers_exact_all_cases"]
            for row in by_seed.values()
        ),
        "locality_violation_area_zero_all_seeds": all(
            row["locality_violation_area_zero_all_cases"]
            for row in by_seed.values()
        ),
        "wall_reduction_positive_all_seeds_operations": all(
            row["wall"]["relative_reduction"] > 0.0
            for seed in by_seed.values()
            for row in seed["operations"].values()
        ),
        "opening_reduction_positive_all_seeds_operations": all(
            row["opening"]["relative_reduction"] > 0.0
            for seed in by_seed.values()
            for row in seed["operations"].values()
        ),
        "wall_paired_ci_positive_all_seeds_operations": all(
            row["wall"]["ci_low"] > 0.0
            for seed in by_seed.values()
            for row in seed["operations"].values()
        ),
        "opening_paired_ci_positive_all_seeds_operations": all(
            row["opening"]["ci_low"] > 0.0
            for seed in by_seed.values()
            for row in seed["operations"].values()
        ),
        "inference_wilcoxon_bonferroni_lt_0_05": maximum_bonferroni_p < 0.05,
    }
    result = {
        "status": "complete" if all(checks.values()) else "needs_iteration",
        "seeds": list(sources),
        "cases": sum(row["cases"] for row in by_seed.values()),
        "sources": {str(seed): path for seed, path in sources.items()},
        "by_seed": by_seed,
        "seed_aggregate": seed_aggregate,
        "checks": checks,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": (
            next(iter(refiner_hashes))
            if len(refiner_hashes) == 1 and None not in refiner_hashes
            else None
        ),
        "room_refinement": "disabled",
        "multiplicity": {
            "method": "Bonferroni",
            "comparison_count": comparison_count,
            "maximum_raw_p": maximum_p,
            "maximum_adjusted_p": maximum_bonferroni_p,
            "familywise_alpha": 0.05,
        },
        "worst_seed_operation_reduction": {
            family: min(
                operation[family]["relative_reduction"]
                for seed in by_seed.values()
                for operation in seed["operations"].values()
            )
            for family in ("wall", "opening")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Vector Residual Inference, Five Seeds",
        "",
        "Room polygons frozen; affected wall/opening proposals refined.",
        "",
        "| Operation | Wall reduction mean [95% CI] | Opening reduction mean [95% CI] |",
        "|---|---:|---:|",
    ]
    for operation, row in seed_aggregate.items():
        lines.append(
            f"| {operation} | {row['wall']['mean_relative_reduction']:.2%} "
            f"[{row['wall']['ci95'][0]:.2%}, {row['wall']['ci95'][1]:.2%}] | "
            f"{row['opening']['mean_relative_reduction']:.2%} "
            f"[{row['opening']['ci95'][0]:.2%}, {row['opening']['ci95'][1]:.2%}] |"
        )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2))
    raise SystemExit(0 if result["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze where and when graph-token flow improves the official score."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 20, 21)
NFES = (1, 2, 4, 8)
COMPONENTS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_grouping_vrel",
    "legalized_mib_vrel",
    "legalized_boundary_vrel",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_hpwl_gap",
    "legalized_rho_mae",
    "legalized_overlap_pairs",
)
COMPLEXITY = (
    "blocks",
    "n_b2b",
    "n_p2b",
    "n_pins",
    "boundary_blocks",
    "cluster_groups",
    "mib_groups",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_component_rows_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def load_rows(path: Path, cache: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "complete":
        raise ValueError("component collection is incomplete")
    rows = payload["rows"]
    store = PackedCaseStore(cache)
    for row in rows:
        index = int(row["index"])
        count = int(store.arrays["n"][index])
        constraints = store.arrays["constraints"][index, :count]
        row["blocks"] = count
        row["n_b2b"] = int(store.arrays["n_b2b"][index])
        row["n_p2b"] = int(store.arrays["n_p2b"][index])
        row["n_pins"] = int(store.arrays["n_pins"][index])
        row["boundary_blocks"] = int(np.count_nonzero(constraints[:, 4]))
        row["cluster_groups"] = int(
            np.unique(constraints[constraints[:, 3] > 0, 3]).size
        )
        row["mib_groups"] = int(
            np.unique(constraints[constraints[:, 2] > 0, 2]).size
        )
        slots = max(1, int(row["legalized_soft_slots"]))
        row["legalized_grouping_vrel"] = row["legalized_grouping"] / slots
        row["legalized_mib_vrel"] = row["legalized_mib"] / slots
        row["legalized_boundary_vrel"] = row["legalized_boundary"] / slots
    return rows


def selected(
    rows: list[dict],
    *,
    arm: str,
    stage: str,
    seed: int | None = None,
    nfe: int | None = None,
) -> list[dict]:
    output = [
        row
        for row in rows
        if row["arm"] == arm and row["stage"] == stage
    ]
    if seed is not None:
        output = [row for row in output if row["seed"] == seed]
    if nfe is not None:
        output = [row for row in output if row["nfe"] == nfe]
    return sorted(output, key=lambda row: int(row["index"]))


def pooled(rows: list[dict], *, arm: str, stage: str, nfe: int | None = None) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for seed in SEEDS:
        for row in selected(rows, arm=arm, stage=stage, seed=seed, nfe=nfe):
            grouped[int(row["index"])].append(row)
    output = []
    for index in sorted(grouped):
        cases = grouped[index]
        averaged = {key: cases[0][key] for key in COMPLEXITY + ("index",)}
        for metric in COMPONENTS:
            averaged[metric] = float(np.mean([row[metric] for row in cases]))
        output.append(averaged)
    return output


def compare(
    candidate: list[dict],
    baseline: list[dict],
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    output = {}
    for index, metric in enumerate(COMPONENTS):
        stats = paired_delta(
            candidate,
            baseline,
            metric,
            iterations=bootstrap,
            seed=seed + index,
            lower_is_better=True,
        )
        candidate_mean = float(
            np.mean([float(row[metric]) for row in candidate])
        )
        baseline_mean = float(
            np.mean([float(row[metric]) for row in baseline])
        )
        stats["candidate_mean"] = candidate_mean
        stats["baseline_mean"] = baseline_mean
        stats["relative_improvement"] = (
            (baseline_mean - candidate_mean) / abs(baseline_mean)
            if baseline_mean
            else None
        )
        output[metric] = stats
    return output


def strata(
    candidate: list[dict],
    baseline: list[dict],
    dimension: str,
) -> dict:
    values = np.asarray([float(row[dimension]) for row in baseline])
    low, high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    bins = {
        "low": np.flatnonzero(values <= low),
        "mid": np.flatnonzero((values > low) & (values <= high)),
        "high": np.flatnonzero(values > high),
    }
    output = {"thresholds": {"low": float(low), "high": float(high)}}
    for name, indices in bins.items():
        if not len(indices):
            output[name] = {"cases": 0}
            continue
        cand = [candidate[int(index)] for index in indices]
        base = [baseline[int(index)] for index in indices]
        cand_cost = float(np.mean([row["legalized_hard_cost"] for row in cand]))
        base_cost = float(np.mean([row["legalized_hard_cost"] for row in base]))
        cand_vrel = float(np.mean([row["legalized_v_rel"] for row in cand]))
        base_vrel = float(np.mean([row["legalized_v_rel"] for row in base]))
        output[name] = {
            "cases": len(indices),
            "candidate_cost": cand_cost,
            "baseline_cost": base_cost,
            "cost_relative_improvement": (base_cost - cand_cost) / base_cost,
            "candidate_v_rel": cand_vrel,
            "baseline_v_rel": base_vrel,
            "v_rel_relative_improvement": (
                (base_vrel - cand_vrel) / base_vrel if base_vrel else None
            ),
        }
    return output


def quantiles(candidate: list[dict], baseline: list[dict]) -> dict:
    levels = (0.50, 0.75, 0.90, 0.95)
    output = {}
    for metric in ("legalized_hard_cost", "legalized_v_rel"):
        baseline_values = np.asarray([row[metric] for row in baseline])
        candidate_values = np.asarray([row[metric] for row in candidate])
        base_quantiles = np.quantile(baseline_values, levels)
        cand_quantiles = np.quantile(candidate_values, levels)
        output[metric] = {
            str(level): {
                "baseline": float(base),
                "candidate": float(cand),
                "relative_improvement": float((base - cand) / abs(base)),
            }
            for level, base, cand in zip(
                levels,
                base_quantiles,
                cand_quantiles,
                strict=True,
            )
        }
    return output


def nfe_curves(rows: list[dict], bootstrap: int) -> dict:
    output = {}
    for nfe in NFES:
        candidate = pooled(rows, arm="candidate", stage="raw", nfe=nfe)
        baseline = pooled(rows, arm="baseline", stage="raw", nfe=nfe)
        comparison = compare(
            candidate,
            baseline,
            bootstrap=bootstrap,
            seed=20262000 + nfe,
        )
        output[str(nfe)] = {
            "candidate_mean": {
                "cost": float(
                    np.mean([row["legalized_hard_cost"] for row in candidate])
                ),
                "v_rel": float(
                    np.mean([row["legalized_v_rel"] for row in candidate])
                ),
            },
            "baseline_mean": {
                "cost": float(
                    np.mean([row["legalized_hard_cost"] for row in baseline])
                ),
                "v_rel": float(
                    np.mean([row["legalized_v_rel"] for row in baseline])
                ),
            },
            "paired": {
                "cost": comparison["legalized_hard_cost"],
                "v_rel": comparison["legalized_v_rel"],
            },
        }
    return output


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, args.cache)
    refined_candidate = pooled(rows, arm="candidate", stage="refine400")
    refined_baseline = pooled(rows, arm="baseline", stage="refine400")
    per_seed = {
        str(seed): compare(
            selected(
                rows,
                arm="candidate",
                stage="refine400",
                seed=seed,
            ),
            selected(
                rows,
                arm="baseline",
                stage="refine400",
                seed=seed,
            ),
            bootstrap=args.bootstrap,
            seed=20261000 + seed * 10,
        )
        for seed in SEEDS
    }
    refined = compare(
        refined_candidate,
        refined_baseline,
        bootstrap=args.bootstrap,
        seed=20261999,
    )
    complexity = {
        dimension: strata(refined_candidate, refined_baseline, dimension)
        for dimension in COMPLEXITY
    }
    tail = quantiles(refined_candidate, refined_baseline)
    curves = nfe_curves(rows, args.bootstrap)
    structural_metrics = (
        "legalized_grouping_vrel",
        "legalized_mib_vrel",
        "legalized_boundary_vrel",
        "legalized_b2b_hpwl_gap",
        "legalized_p2b_hpwl_gap",
    )
    structural_support = [
        metric
        for metric in structural_metrics
        if refined[metric]["mean_delta"] < 0.0
        and refined[metric]["ci_high"] < 0.0
    ]
    def complex_beats_simple(result: dict, metric: str) -> bool:
        high = result.get("high", {}).get(metric)
        low = result.get("low", {}).get(metric)
        return high is not None and low is not None and high > low + 0.01

    complexity_support = {
        dimension: {
            "cost": complex_beats_simple(result, "cost_relative_improvement"),
            "v_rel": complex_beats_simple(
                result,
                "v_rel_relative_improvement",
            ),
        }
        for dimension, result in complexity.items()
    }
    tail_support = {
        metric: (
            detail["0.95"]["relative_improvement"]
            > detail["0.5"]["relative_improvement"] + 0.01
        )
        for metric, detail in tail.items()
    }
    low_nfe_advantage = min(
        curves[str(nfe)]["paired"]["cost"]["relative_delta"]
        for nfe in (1, 2)
    )
    high_nfe_advantage = curves["8"]["paired"]["cost"]["relative_delta"]
    checks = {
        "structural_component_supported": bool(structural_support),
        "complexity_heterogeneity_supported": any(
            any(flags.values()) for flags in complexity_support.values()
        ),
        "tail_improvement_supported": any(tail_support.values()),
        "low_nfe_advantage_supported": (
            low_nfe_advantage < high_nfe_advantage - 0.01
        ),
    }
    report = {
        "status": "supported" if any(checks.values()) else "not_supported",
        "scope": "matched three-seed official-100 mechanism diagnostics",
        "definitions": {
            "v_rel": (
                "grouping + MIB shape mismatch + boundary violation, "
                "divided by available soft slots"
            ),
            "no_opening_metric": (
                "FloorSet does not expose floorplan openings; boundary and p2b "
                "are reported as structural proxies, not relabeled as openings."
            ),
        },
        "refined_pooled": refined,
        "refined_per_seed": per_seed,
        "complexity_strata": complexity,
        "complexity_support": complexity_support,
        "tail_quantiles": tail,
        "tail_support": tail_support,
        "nfe_curves": curves,
        "structural_support": structural_support,
        "checks": checks,
        "multiple_comparison_note": (
            "Component intervals are exploratory; no claim survives a formal "
            "multiplicity correction unless separately stated."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

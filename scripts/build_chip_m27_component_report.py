#!/usr/bin/env python3
"""Compare canonical M27 against scratch and Trinity across score components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
CASE_KEY = "nfe4_refine1600"
METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_mib",
    "legalized_rho_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-cases", type=Path, required=True)
    parser.add_argument("--scratch-cases", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    payload = load(path)
    if CASE_KEY in payload:
        return payload[CASE_KEY]
    return payload["rows"]


def compare(
    candidate: list[dict],
    baseline: list[dict],
    *,
    seed: int,
) -> dict:
    output = {}
    for index, metric in enumerate(METRICS):
        stats = paired_delta(
            candidate,
            baseline,
            metric,
            iterations=10000,
            seed=seed + index,
            lower_is_better=True,
        )
        candidate_mean = float(np.mean([row[metric] for row in candidate]))
        baseline_mean = float(np.mean([row[metric] for row in baseline]))
        stats["candidate_mean"] = candidate_mean
        stats["baseline_mean"] = baseline_mean
        stats["relative_improvement"] = (
            (baseline_mean - candidate_mean) / abs(baseline_mean)
            if baseline_mean
            else None
        )
        output[metric] = stats
    return output


def quantiles(candidate: list[dict], baseline: list[dict]) -> dict:
    output = {}
    for metric in (
        "legalized_hard_cost",
        "legalized_v_rel",
        "legalized_p2b_hpwl_gap",
        "legalized_area_gap",
    ):
        base = np.asarray([row[metric] for row in baseline])
        cand = np.asarray([row[metric] for row in candidate])
        base_values = np.quantile(base, (0.5, 0.75, 0.9, 0.95))
        cand_values = np.quantile(cand, (0.5, 0.75, 0.9, 0.95))
        output[metric] = {
            str(level): {
                "baseline": float(b),
                "candidate": float(c),
                "relative_improvement": float((b - c) / abs(b)) if b else None,
            }
            for level, b, c in zip(
                (0.5, 0.75, 0.9, 0.95),
                base_values,
                cand_values,
                strict=True,
            )
        }
    return output


def complexity_strata(
    candidate: list[dict],
    baseline: list[dict],
    store: PackedCaseStore,
) -> dict:
    by_index = {int(row["index"]): row for row in candidate}
    base_index = {int(row["index"]): row for row in baseline}
    strata = {
        "21-40": [],
        "41-80": [],
        "81-100": [],
        "101-120": [],
    }
    for index, cand in by_index.items():
        count = int(store.arrays["n"][index])
        if count <= 40:
            name = "21-40"
        elif count <= 80:
            name = "41-80"
        elif count <= 100:
            name = "81-100"
        else:
            name = "101-120"
        strata[name].append((cand, base_index[index]))
    output = {}
    for name, pairs in strata.items():
        if not pairs:
            continue
        cand = [pair[0] for pair in pairs]
        base = [pair[1] for pair in pairs]
        cand_cost = float(np.mean([row["legalized_hard_cost"] for row in cand]))
        base_cost = float(np.mean([row["legalized_hard_cost"] for row in base]))
        cand_v_rel = float(np.mean([row["legalized_v_rel"] for row in cand]))
        base_v_rel = float(np.mean([row["legalized_v_rel"] for row in base]))
        output[name] = {
            "cases": len(pairs),
            "cost_relative_improvement": (base_cost - cand_cost) / base_cost,
            "v_rel_relative_improvement": (
                (base_v_rel - cand_v_rel) / base_v_rel
                if base_v_rel
                else None
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    candidate_report = load(args.candidate)
    candidate = rows(args.candidate_cases)
    scratch = rows(args.scratch_cases)
    anchor = rows(args.anchor)
    store = PackedCaseStore(args.cache)
    candidate_vs_scratch = compare(candidate, scratch, seed=20280500)
    candidate_vs_anchor = compare(candidate, anchor, seed=20280600)
    report = {
        "version": "chip-m27-component-evidence-v1",
        "status": "complete_descriptive",
        "candidate": str(args.candidate),
        "candidate_summary": candidate_report["results"][CASE_KEY],
        "candidate_vs_scratch": candidate_vs_scratch,
        "candidate_vs_anchor": candidate_vs_anchor,
        "candidate_vs_scratch_quantiles": quantiles(candidate, scratch),
        "candidate_vs_anchor_quantiles": quantiles(candidate, anchor),
        "candidate_vs_scratch_complexity": complexity_strata(
            candidate,
            scratch,
            store,
        ),
        "candidate_vs_anchor_complexity": complexity_strata(
            candidate,
            anchor,
            store,
        ),
        "claim_boundary": (
            "One canonical training seed at fixed inference seed19. "
            "Training seed/order sensitivity remains a known limitation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_vs_scratch": {
                    key: value["relative_improvement"]
                    for key, value in candidate_vs_scratch.items()
                },
                "candidate_vs_anchor": {
                    key: value["relative_improvement"]
                    for key, value in candidate_vs_anchor.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

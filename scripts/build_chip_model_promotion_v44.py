#!/usr/bin/env python3
"""Select typed hierarchy or graph-token flow at the frozen refiner budget."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 20, 21)
METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_area_gap",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def result_key(report: dict) -> str:
    keys = sorted(report["results"])
    if len(keys) != 1:
        raise ValueError(f"expected one result setting, found {keys}")
    return keys[0]


def metrics(report: dict) -> dict[str, float]:
    row = report["results"][result_key(report)]
    return {metric: float(row[metric]) for metric in METRICS}


def main() -> None:
    by_arm = {}
    rows_by_arm = {}
    for arm, directory, prefix in (
        ("typed_hierarchy", "refiner_official", "candidate"),
        ("graph_token", "control_official", "control"),
    ):
        reports = {}
        rows = {}
        for seed in SEEDS:
            path = (
                ROOT
                / f"outputs/chip_hierarchy/v44/{directory}"
                / f"{prefix}_boundary2_detglop_1600_seed{seed}.json"
            )
            reports[str(seed)] = metrics(read(path))
            case_path = path.with_name(f"{path.stem}_cases.json")
            rows[seed] = read(case_path)["nfe4_refine1600"]
        by_arm[arm] = reports
        rows_by_arm[arm] = rows
    seed_results = {}
    for seed in SEEDS:
        paired = {
            metric: paired_delta(
                rows_by_arm["graph_token"][seed],
                rows_by_arm["typed_hierarchy"][seed],
                metric,
                iterations=10000,
                seed=20271000 + seed * 20 + index,
                lower_is_better=True,
            )
            for index, metric in enumerate(METRICS)
        }
        seed_results[str(seed)] = {
            "typed_hierarchy": by_arm["typed_hierarchy"][str(seed)],
            "graph_token": by_arm["graph_token"][str(seed)],
            "relative_graph_token_vs_typed": {
                metric: by_arm["graph_token"][str(seed)][metric]
                / by_arm["typed_hierarchy"][str(seed)][metric]
                - 1.0
                for metric in METRICS
            },
            "paired": paired,
        }
    pooled = {
        arm: {
            metric: float(
                np.mean([by_arm[arm][str(seed)][metric] for seed in SEEDS])
            )
            for metric in METRICS
        }
        for arm in by_arm
    }
    checks = {
        "graph_token_better_cost_mean": (
            pooled["graph_token"]["legalized_hard_cost"]
            < pooled["typed_hierarchy"]["legalized_hard_cost"]
        ),
        "graph_token_better_v_rel_mean": (
            pooled["graph_token"]["legalized_v_rel"]
            < pooled["typed_hierarchy"]["legalized_v_rel"]
        ),
        "graph_token_better_cost_all_seeds": all(
            by_arm["graph_token"][str(seed)]["legalized_hard_cost"]
            < by_arm["typed_hierarchy"][str(seed)]["legalized_hard_cost"]
            for seed in SEEDS
        ),
        "graph_token_better_v_rel_all_seeds": all(
            by_arm["graph_token"][str(seed)]["legalized_v_rel"]
            < by_arm["typed_hierarchy"][str(seed)]["legalized_v_rel"]
            for seed in SEEDS
        ),
    }
    point_estimate = (
        "graph_token"
        if checks["graph_token_better_cost_mean"]
        and checks["graph_token_better_v_rel_mean"]
        else "typed_hierarchy"
    )
    promoted = point_estimate if all(checks.values()) else "no_uniform_winner"
    report = {
        "status": "pass" if all(checks.values()) else "mixed",
        "scope": "Three seeds, Official-100, identical deterministic 1600-step refiner",
        "promoted_architecture": promoted,
        "point_estimate_architecture": point_estimate,
        "pooled_seed_mean": pooled,
        "relative_graph_token_vs_typed": {
            metric: pooled["graph_token"][metric]
            / pooled["typed_hierarchy"][metric]
            - 1.0
            for metric in METRICS
        },
        "seeds": seed_results,
        "checks": checks,
        "claim_boundary": (
            "Promote graph_token if it is better on both headline metrics. "
            "The typed hierarchy remains a negative architecture result."
        ),
    }
    output = ROOT / "outputs/reports/chip_model_promotion_v44.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "promoted_architecture": promoted,
                "checks": checks,
                "pooled_seed_mean": pooled,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

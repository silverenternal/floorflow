#!/usr/bin/env python3
"""Aggregate the preregistered refiner development sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import ttest_rel

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_b2b_hpwl_gap",
    "legalized_area_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_feasible_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/v44/refiner_screen",
    )
    parser.add_argument(
        "--default-arm",
        default="candidate_default_lp2s_dev.json",
    )
    parser.add_argument(
        "--control-arm",
        default="baseline_seed19_dev.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_refiner_sota_sweep_v44.json",
    )
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def result_key(path: Path) -> str:
    keys = sorted(read(path)["results"])
    if len(keys) != 1:
        raise ValueError(f"expected one result setting in {path}, found {keys}")
    return keys[0]


def aligned_rows(path: Path, indices: list[int]) -> list[dict]:
    cases = read(path.with_name(f"{path.stem}_cases.json"))[result_key(path)]
    by_index = {int(row["index"]): row for row in cases}
    return [by_index[index] for index in indices]


def relative(first: float, baseline: float) -> float | None:
    return first / baseline - 1.0 if baseline else None


def complete_summary(summary: dict, cases: list[dict]) -> dict[str, float]:
    return {
        metric: float(
            summary[metric]
            if metric in summary
            else sum(float(row[metric]) for row in cases) / len(cases)
        )
        for metric in METRICS
    }


def fdr(values: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(values, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 1.0
    count = len(ordered)
    for rank in range(count, 0, -1):
        name, p_value = ordered[rank - 1]
        running = min(running, p_value * count / rank)
        adjusted[name] = running
    return adjusted


def main() -> None:
    args = parse_args()
    default_path = args.directory / args.default_arm
    control_path = args.directory / args.control_arm
    default = read(default_path)
    control = read(control_path)
    candidate_paths = sorted(
        path
        for path in args.directory.glob("*_dev.json")
        if path.name not in {args.default_arm, args.control_arm}
    )
    protocol_timeout = default.get("legalizer_time_limit_ms")
    protocol_deterministic = default.get("deterministic")
    report_paths = []
    excluded_protocols = []
    for path in candidate_paths:
        candidate_timeout = read(path).get("legalizer_time_limit_ms")
        result = read(path)
        if (
            candidate_timeout == protocol_timeout
            and result.get("deterministic") == protocol_deterministic
        ):
            report_paths.append(path)
        else:
            excluded_protocols.append(
                {
                    "arm": path.stem,
                    "legalizer_time_limit_ms": candidate_timeout,
                    "deterministic": result.get("deterministic"),
                }
            )
    default_key = result_key(default_path)
    default_cases = read(
        default_path.with_name(f"{default_path.stem}_cases.json")
    )[default_key]
    indices = [int(row["index"]) for row in default_cases]
    default_rows = aligned_rows(default_path, indices)
    control_key = result_key(control_path)
    control_cases = read(
        control_path.with_name(f"{control_path.stem}_cases.json")
    )[control_key]
    control_summary = complete_summary(
        control["results"][control_key],
        control_cases,
    )
    default_summary = complete_summary(default["results"][default_key], default_rows)
    arms = []
    paired_p_values = []
    for index, path in enumerate(report_paths):
        result = read(path)
        key = result_key(path)
        cases = aligned_rows(path, indices)
        summary = complete_summary(result["results"][key], cases)
        paired = paired_delta(
            cases,
            default_rows,
            "legalized_hard_cost",
            iterations=4000,
            seed=20264400 + index,
            lower_is_better=True,
        )
        p_value = float(
            ttest_rel(
                [row["legalized_hard_cost"] for row in cases],
                [row["legalized_hard_cost"] for row in default_rows],
                alternative="less",
            ).pvalue
        )
        paired_p_values.append((path.stem, p_value))
        wins_cost = sum(
            row["legalized_hard_cost"] < base["legalized_hard_cost"]
            for row, base in zip(cases, default_rows, strict=True)
        )
        wins_v_rel = sum(
            row["legalized_v_rel"] < base["legalized_v_rel"]
            for row, base in zip(cases, default_rows, strict=True)
        )
        arms.append(
            {
                "arm": path.stem,
                "result_key": key,
                "report": str(path.relative_to(ROOT)),
                "refiner": result.get("refiner"),
                "summary": summary,
                "relative_to_candidate_default": {
                    metric: relative(float(summary[metric]), float(default_summary[metric]))
                    for metric in METRICS
                },
                "relative_to_seed19_control": {
                    metric: relative(float(summary[metric]), float(control_summary[metric]))
                    for metric in METRICS
                },
                "paired_cost_vs_candidate_default": paired,
                "paired_cost_t_p_value_less": p_value,
                "case_wins": {
                    "cost": wins_cost,
                    "v_rel": wins_v_rel,
                    "cases": len(cases),
                },
                "both_metrics_beat_candidate_default": (
                    float(summary["legalized_hard_cost"])
                    < float(default_summary["legalized_hard_cost"])
                    and float(summary["legalized_v_rel"])
                    < float(default_summary["legalized_v_rel"])
                ),
                "legalized_feasible_fraction": float(
                    summary["legalized_feasible_fraction"]
                ),
            }
        )
    adjusted = fdr(paired_p_values)
    for arm in arms:
        arm["paired_cost_bh_q_value"] = adjusted[arm["arm"]]
    eligible = [
        arm
        for arm in arms
        if arm["both_metrics_beat_candidate_default"]
        and arm["legalized_feasible_fraction"] == 1.0
    ]
    best = min(
        eligible or arms,
        key=lambda arm: (
            arm["summary"]["legalized_hard_cost"],
            arm["summary"]["legalized_v_rel"],
        ),
    )
    report = {
        "status": "pass" if eligible else "reject_refiner_sweep",
        "manifest": "outputs/reports/chip_refiner_screen_manifest_v44.json",
        "case_indices": indices,
        "default_arm": default_path.name,
        "control_arm": control_path.name,
        "default_summary": default_summary,
        "control_summary": control_summary,
        "attempted_arms": len(arms) + 2,
        "excluded_different_legalizer_budget": excluded_protocols,
        "arms": arms,
        "eligible_both_metric_arms": [arm["arm"] for arm in eligible],
        "best_primary": best["arm"],
        "promotion_rule": (
            "Promote only arms that beat the matched candidate default on both cost "
            "and V_rel with legalized feasible fraction 1.0. Confirm the frozen lead "
            "on Official-100; do not select from Official-100."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "attempted_arms": report["attempted_arms"],
                "best_primary": report["best_primary"],
                "eligible": report["eligible_both_metric_arms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

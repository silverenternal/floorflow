#!/usr/bin/env python3
"""Aggregate a three-seed official-100 refiner candidate against Trinity."""

from __future__ import annotations

import argparse
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
    "legalized_rho_mae",
    "legalized_mib",
    "legalized_overlap_pairs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument(
        "--report-template",
        default="candidate_boundary2_detglop_{steps}_seed{seed}.json",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs/chip_hierarchy/v44/refiner_official",
    )
    parser.add_argument("--allow-no-projector", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }


def main() -> None:
    args = parse_args()
    output = args.output or (
        ROOT
        / f"outputs/reports/chip_refiner_official_{args.steps}_three_seed_v44.json"
    )
    trinity = read(
        ROOT / "outputs/reports/chip_trinity_d192_component_gap_v1.json"
    )
    trinity_anchor = trinity["main_comparison"]["trinity"]
    thresholds = {
        "cost": float(trinity_anchor["mean_cost"]),
        "v_rel": float(trinity_anchor["mean_v_rel"]),
    }
    candidate_rows = {}
    candidate_reports = {}
    for seed in SEEDS:
        report_path = (
            ROOT
                / args.report_dir
                / args.report_template.format(steps=args.steps, seed=seed)
        )
        cases_path = report_path.with_name(f"{report_path.stem}_cases.json")
        report = read(report_path)
        key = f"nfe4_refine{args.steps}"
        if key not in report["results"]:
            raise ValueError(f"missing {key} in {report_path}")
        if report.get("deterministic") is not True:
            raise ValueError(f"non-deterministic report: {report_path}")
        if int(report.get("legalizer_time_limit_ms", -1)) != 2000:
            raise ValueError(f"wrong legalizer budget: {report_path}")
        projector_iterations = report.get("projector_iterations")
        if not args.allow_no_projector and int(projector_iterations or -1) != 8:
            raise ValueError(f"wrong projector iterations: {report_path}")
        refiner = report.get("refiner", {})
        if float(refiner.get("weights", {}).get("boundary", -1.0)) != args.boundary_weight:
            raise ValueError(f"wrong boundary weight: {report_path}")
        candidate_rows[seed] = read(cases_path)[key]
        candidate_reports[seed] = report
    trinity_rows = {
        seed: read(
            ROOT / f"outputs/reports/trinity_d192_component_seed{seed}_v1.json"
        )["rows"]
        for seed in SEEDS
    }
    seed_results = {}
    for seed in SEEDS:
        current = mean_metrics(candidate_rows[seed])
        paired_cost = paired_delta(
            candidate_rows[seed],
            trinity_rows[seed],
            "legalized_hard_cost",
            iterations=10000,
            seed=20265000 + seed,
            lower_is_better=True,
        )
        paired_v_rel = paired_delta(
            candidate_rows[seed],
            trinity_rows[seed],
            "legalized_v_rel",
            iterations=10000,
            seed=20266000 + seed,
            lower_is_better=True,
        )
        seed_results[str(seed)] = {
            "candidate": current,
            "trinity": {
                "cost": float(
                    np.mean(
                        [float(row["legalized_hard_cost"]) for row in trinity_rows[seed]]
                    )
                ),
                "v_rel": float(
                    np.mean(
                        [float(row["legalized_v_rel"]) for row in trinity_rows[seed]]
                    )
                ),
            },
            "gap_to_trinity": {
                "cost_relative": current["legalized_hard_cost"] / thresholds["cost"] - 1.0,
                "v_rel_relative": current["legalized_v_rel"] / thresholds["v_rel"] - 1.0,
            },
            "beats_both": (
                current["legalized_hard_cost"] < thresholds["cost"]
                and current["legalized_v_rel"] < thresholds["v_rel"]
            ),
            "paired_cost": paired_cost,
            "paired_v_rel": paired_v_rel,
            "cost_wins_ties_losses": [
                int(
                    np.sum(
                        np.asarray(
                            [
                                float(row["legalized_hard_cost"])
                                - float(reference["cost"])
                                for row, reference in zip(
                                    candidate_rows[seed],
                                    trinity_rows[seed],
                                    strict=True,
                                )
                            ]
                        )
                        < 0.0
                    )
                ),
                int(
                    np.sum(
                        np.asarray(
                            [
                                float(row["legalized_hard_cost"])
                                - float(reference["cost"])
                                for row, reference in zip(
                                    candidate_rows[seed],
                                    trinity_rows[seed],
                                    strict=True,
                                )
                            ]
                        )
                        == 0.0
                    )
                ),
                int(
                    np.sum(
                        np.asarray(
                            [
                                float(row["legalized_hard_cost"])
                                - float(reference["cost"])
                                for row, reference in zip(
                                    candidate_rows[seed],
                                    trinity_rows[seed],
                                    strict=True,
                                )
                            ]
                        )
                        > 0.0
                    )
                ),
            ],
            "legalized_feasible_fraction": float(
                candidate_reports[seed]["results"][
                    f"nfe4_refine{args.steps}"
                ]["legalized_feasible_fraction"]
            ),
        }
    all_rows = [
        row for seed in SEEDS for row in candidate_rows[seed]
    ]
    all_trinity = [
        row for seed in SEEDS for row in trinity_rows[seed]
    ]
    candidate_mean = mean_metrics(all_rows)
    checks = {
        "all_seed_legality": all(
            result["legalized_feasible_fraction"] == 1.0
            for result in seed_results.values()
        ),
        "all_seeds_beat_cost": all(
            result["candidate"]["legalized_hard_cost"] < thresholds["cost"]
            for result in seed_results.values()
        ),
        "all_seeds_beat_v_rel": all(
            result["candidate"]["legalized_v_rel"] < thresholds["v_rel"]
            for result in seed_results.values()
        ),
        "seed_mean_beats_cost": (
            candidate_mean["legalized_hard_cost"] < thresholds["cost"]
        ),
        "seed_mean_beats_v_rel": (
            candidate_mean["legalized_v_rel"] < thresholds["v_rel"]
        ),
    }
    report = {
        "status": "pass_three_seed_sota_anchor" if all(checks.values()) else "rejected",
        "status_detail": (
            "candidate beats the fixed 400-step Trinity d192 anchor; "
            "this is not a matched-compute result when candidate_steps != 400"
        ),
        "comparison_budget": {
            "candidate_refiner_steps": args.steps,
            "anchor_refiner_steps": 400,
            "matched": args.steps == 400,
        },
        "scope": (
            f"Official-100, NFE4 + boundary={args.boundary_weight:g} + "
            f"{args.steps}-step linear refiner, deterministic, LP=2000 ms"
        ),
        "thresholds": thresholds,
        "candidate_seed_mean": candidate_mean,
        "gap_to_trinity": {
            "cost_relative": candidate_mean["legalized_hard_cost"] / thresholds["cost"] - 1.0,
            "v_rel_relative": candidate_mean["legalized_v_rel"] / thresholds["v_rel"] - 1.0,
        },
        "pooled_paired_cost": paired_delta(
            all_rows,
            all_trinity,
            "legalized_hard_cost",
            iterations=10000,
            seed=20267001,
            lower_is_better=True,
        ),
        "pooled_paired_v_rel": paired_delta(
            all_rows,
            all_trinity,
            "legalized_v_rel",
            iterations=10000,
            seed=20267002,
            lower_is_better=True,
        ),
        "seeds": seed_results,
        "component_comparison": {
            metric: {
                "candidate": candidate_mean[metric],
                "trinity": float(np.mean([float(row[metric]) for row in all_trinity])),
                "relative_delta": (
                    candidate_mean[metric]
                    / float(np.mean([float(row[metric]) for row in all_trinity]))
                    - 1.0
                    if float(np.mean([float(row[metric]) for row in all_trinity]))
                    else None
                ),
                "paired": paired_delta(
                    all_rows,
                    all_trinity,
                    metric,
                    iterations=10000,
                    seed=20268000 + index,
                    lower_is_better=True,
                ),
            }
            for index, metric in enumerate(METRICS)
        },
        "checks": checks,
        "evidence": {
            "candidate_reports": [
                str(
                    (
                        ROOT
                        / args.report_dir
                        / args.report_template.format(steps=args.steps, seed=seed)
                    ).relative_to(ROOT)
                )
                for seed in SEEDS
            ],
            "trinity_reports": [
                f"outputs/reports/trinity_d192_component_seed{seed}_v1.json"
                for seed in SEEDS
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate_seed_mean": candidate_mean,
                "gap_to_trinity": report["gap_to_trinity"],
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

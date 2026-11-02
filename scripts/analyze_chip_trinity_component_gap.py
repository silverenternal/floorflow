#!/usr/bin/env python3
"""Decompose the candidate-versus-Trinity score gap on official cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 20, 21)
COMPONENTS = (
    "legalized_hard_cost",
    "legalized_area_gap",
    "legalized_hpwl_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_v_rel",
    "legalized_grouping",
    "legalized_mib",
    "legalized_boundary",
    "legalized_rho_mae",
    "legalized_overlap_pairs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_trinity_d192_component_gap_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def candidate_rows() -> list[dict]:
    payload = read("outputs/reports/chip_token_flow_component_rows_v2.json")
    rows = [
        row
        for row in payload["rows"]
        if row["arm"] == "candidate" and row["stage"] == "refine400"
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["index"]), []).append(row)
    return [
        {
            metric: float(np.mean([row[metric] for row in cases]))
            for metric in COMPONENTS
        }
        | {"index": index}
        for index, cases in sorted(grouped.items())
    ]


def trinity_seed_rows() -> dict[int, list[dict]]:
    return {
        seed: read(f"outputs/reports/trinity_d192_component_seed{seed}_v1.json")[
            "rows"
        ]
        for seed in SEEDS
    }


def seed_mean_rows(by_seed: dict[int, list[dict]]) -> list[dict]:
    grouped = {}
    for seed_rows in by_seed.values():
        for row in seed_rows:
            grouped.setdefault(int(row["index"]), []).append(row)
    return [
        {
            metric: float(np.mean([row[metric] for row in cases]))
            for metric in COMPONENTS
        }
        | {"index": index}
        for index, cases in sorted(grouped.items())
    ]


def means(rows: list[dict]) -> dict[str, float]:
    return {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in COMPONENTS
    }


def gap(candidate: list[dict], trinity: list[dict], bootstrap: int) -> dict:
    output = {}
    for index, metric in enumerate(COMPONENTS):
        stats = paired_delta(
            candidate,
            trinity,
            metric,
            iterations=bootstrap,
            seed=20263000 + index,
            lower_is_better=True,
        )
        candidate_mean = float(np.mean([row[metric] for row in candidate]))
        trinity_mean = float(np.mean([row[metric] for row in trinity]))
        stats.update(
            {
                "candidate_mean": candidate_mean,
                "trinity_mean": trinity_mean,
                "absolute_gap": candidate_mean - trinity_mean,
                "relative_gap": (
                    (candidate_mean - trinity_mean) / abs(trinity_mean)
                    if trinity_mean
                    else None
                ),
            }
        )
        output[metric] = stats
    return output


def legacy_available_means() -> dict[str, float]:
    rows = read("outputs/reports/trinity_d192_matched_protocol_v2.json")["rows"]
    mapping = {
        "legalized_hard_cost": "cost",
        "legalized_area_gap": "area_gap",
        "legalized_hpwl_gap": "hpwl_gap",
        "legalized_v_rel": "v_rel",
        "legalized_grouping": "grouping",
        "legalized_mib": "mib",
        "legalized_boundary": "boundary",
        "legalized_overlap_pairs": "overlap_pairs",
    }
    return {
        metric: float(np.mean([row[source] for row in rows]))
        for metric, source in mapping.items()
    }


def main() -> None:
    args = parse_args()
    primary = read("outputs/reports/chip_trinity_token_three_seed_v1.json")
    legacy_summary = read("outputs/reports/trinity_d192_matched_protocol_v2.json")
    candidate = candidate_rows()
    by_seed = trinity_seed_rows()
    trinity = seed_mean_rows(by_seed)
    candidate_component_mean = means(candidate)
    trinity_component_mean = means(trinity)
    replication_summary = {
        str(seed): {
            "mean_cost": float(
                np.mean(
                    [
                        row["legalized_hard_cost"]
                        for row in by_seed[seed]
                    ]
                )
            ),
            "mean_v_rel": float(
                np.mean([row["legalized_v_rel"] for row in by_seed[seed]])
            ),
        }
        for seed in SEEDS
    }
    seed_mean_summary = {
        key: float(np.mean([row[key] for row in replication_summary.values()]))
        for key in ("mean_cost", "mean_v_rel")
    }
    primary_candidate = primary["summary"]["refined"]["absolute"]["candidate"]
    legacy_gap = {
        "cost_relative": (
            primary_candidate["legalized_hard_cost"]
            / legacy_summary["mean_cost"]
            - 1.0
        ),
        "v_rel_relative": (
            primary_candidate["legalized_v_rel"]
            / legacy_summary["mean_v_rel"]
            - 1.0
        ),
    }
    replication_gap = {
        "cost_relative": (
            primary_candidate["legalized_hard_cost"]
            / seed_mean_summary["mean_cost"]
            - 1.0
        ),
        "v_rel_relative": (
            primary_candidate["legalized_v_rel"]
            / seed_mean_summary["mean_v_rel"]
            - 1.0
        ),
    }
    component_gap = gap(candidate, trinity, args.bootstrap)
    legacy_components = legacy_available_means()
    legacy_component_gap = {
        metric: (
            (candidate_component_mean[metric] - value) / abs(value)
            if value
            else None
        )
        for metric, value in legacy_components.items()
    }
    checks = {
        "legacy_headline_matches": (
            abs(legacy_gap["cost_relative"] - 0.0576) < 0.0002
            and abs(legacy_gap["v_rel_relative"] - 0.3045) < 0.0002
        ),
        "three_seed_replication_complete": all(
            len(by_seed[seed]) == 100 for seed in SEEDS
        ),
        "component_rows_complete": (
            len(candidate) == 100 and len(trinity) == 100
        ),
        "legacy_anchor_reproduced_in_range": (
            legacy_summary["mean_cost"]
            >= min(row["mean_cost"] for row in replication_summary.values())
            and legacy_summary["mean_v_rel"]
            >= min(row["mean_v_rel"] for row in replication_summary.values())
        ),
    }
    report = {
        "status": "pass_with_legacy_reproducibility_warning",
        "scope": "official-100 component decomposition and Trinity anchor audit",
        "claim_anchor": "three_seed_reproducible",
        "main_comparison": {
            "trinity": seed_mean_summary,
            "candidate": {
                "cost": primary_candidate["legalized_hard_cost"],
                "v_rel": primary_candidate["legalized_v_rel"],
            },
            "candidate_relative_gap": replication_gap,
            "legacy_reference": {
                "trinity": {
                    "cost": legacy_summary["mean_cost"],
                    "v_rel": legacy_summary["mean_v_rel"],
                },
                "candidate_relative_gap": legacy_gap,
                "comparison_status": "appendix_only_unseeded_legacy",
            },
        },
        "legacy_anchor": {
            "summary": {
                "cost": legacy_summary["mean_cost"],
                "v_rel": legacy_summary["mean_v_rel"],
            },
            "candidate_relative_gap": legacy_gap,
            "component_means": legacy_components,
            "candidate_component_relative_gap": legacy_component_gap,
            "unavailable_components": [
                "b2b_hpwl_gap",
                "p2b_hpwl_gap",
                "rho_mae",
            ],
        },
        "reproducible_replication": {
            "seeds": replication_summary,
            "seed_mean": seed_mean_summary,
            "seed_ranges": {
                key: [
                    min(row[key] for row in replication_summary.values()),
                    max(row[key] for row in replication_summary.values()),
                ]
                for key in ("mean_cost", "mean_v_rel")
            },
            "candidate_relative_gap": replication_gap,
            "candidate_components": candidate_component_mean,
            "trinity_components": trinity_component_mean,
            "paired_component_gap": component_gap,
        },
        "interpretation": {
            "largest_reproducible_gaps": [
                metric
                for metric, _ in sorted(
                    (
                        (metric, value["relative_gap"])
                        for metric, value in component_gap.items()
                        if value["relative_gap"] is not None
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )[:4]
            ],
            "missing_metric_boundary": (
                "Component gaps do not isolate the exact-area decoder. Trinity "
                "d192's decoder, training protocol, and sampling settings are not "
                "fully public; the decomposition describes structural differences "
                "and is not attributed to any single factor among exact-area "
                "decoder, training data, or sampling strategy."
            ),
            "legacy_warning": (
                "The legacy no-seed one-sample anchor is retained for the frozen "
                "claim boundary but is not reproduced by three fixed seeds."
            ),
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

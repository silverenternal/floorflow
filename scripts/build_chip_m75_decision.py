#!/usr/bin/env python3
"""Compare the M75 seed-diverse selector with all three Trinity anchors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta
from scripts.build_chip_m74_seed_report import (
    METRICS,
    TRINITY,
    seed_summary,
    trinity_rows,
)

CANDIDATE = "outputs/reports/chip_m75_seed_diverse_guarded_official100_v1.json"
M74 = "outputs/reports/chip_m74_cross_checkpoint_seed_report_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=ROOT / CANDIDATE)
    parser.add_argument("--m74", type=Path, default=ROOT / M74)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m75_decision_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def cluster_paired(
    candidates: dict[int, list[dict]],
    references: dict[int, list[dict]],
    iterations: int,
) -> dict:
    result = {}
    for offset, metric in enumerate(METRICS):
        current = [
            {
                "index": index,
                "value": sum(
                    float(candidates[seed][index][metric]) for seed in candidates
                )
                / len(candidates),
            }
            for index in range(len(candidates[19]))
        ]
        baseline = [
            {
                "index": index,
                "value": sum(
                    float(references[seed][index][metric]) for seed in references
                )
                / len(references),
            }
            for index in range(len(references[19]))
        ]
        result[metric] = paired_delta(
            current,
            baseline,
            "value",
            iterations=iterations,
            seed=7600 + offset,
            lower_is_better=True,
        )
    return result


def main() -> None:
    args = parse_args()
    candidate_report = load(args.candidate)
    candidate = [
        {
            "index": int(row["index"]),
            **{key: float(row[key]) for key in METRICS},
            "feasible": bool(row["feasible"]),
        }
        for row in candidate_report["selected_rows"]
    ]
    seeds = {}
    pooled_candidate = []
    pooled_reference = []
    candidates_by_seed = {}
    references_by_seed = {}
    for seed in (19, 20, 21):
        reference = trinity_rows(TRINITY[seed])
        candidates_by_seed[seed] = candidate
        references_by_seed[seed] = reference
        summary = seed_summary(
            candidate,
            reference,
            args.bootstrap,
            seed,
        )
        summary["selection_counts"] = candidate_report["selection_counts"]
        seeds[str(seed)] = summary
        for offset, (current, baseline) in enumerate(
            zip(candidate, reference, strict=True)
        ):
            unique_index = seed * 1000 + offset
            pooled_candidate.append({**current, "index": unique_index})
            pooled_reference.append({**baseline, "index": unique_index})
    pooled = seed_summary(
        pooled_candidate,
        pooled_reference,
        args.bootstrap,
        75,
    )
    case_cluster = cluster_paired(
        candidates_by_seed,
        references_by_seed,
        args.bootstrap,
    )
    relative_means = {
        seed: values["relative_means"] for seed, values in seeds.items()
    }
    cost_mean_wins = sum(values["cost"] < 0.0 for values in relative_means.values())
    v_rel_mean_wins = sum(values["v_rel"] < 0.0 for values in relative_means.values())
    checks = {
        "candidate_feasibility_one": all(row["feasible"] for row in candidate),
        "cost_mean_better_against_all_three_anchors": cost_mean_wins == 3,
        "v_rel_mean_better_against_all_three_anchors": v_rel_mean_wins == 3,
        "pooled_cost_ci_excludes_zero": case_cluster["cost"]["ci_high"] < 0.0,
        "pooled_v_rel_ci_excludes_zero": case_cluster["v_rel"]["ci_high"] < 0.0,
        "pooled_area_ci_excludes_zero": case_cluster["area_gap"]["ci_high"] < 0.0,
        "pooled_b2b_ci_excludes_zero": (
            case_cluster["b2b_hpwl_gap"]["ci_high"] < 0.0
        ),
    }
    stable_primary = (
        checks["candidate_feasibility_one"]
        and checks["cost_mean_better_against_all_three_anchors"]
        and checks["v_rel_mean_better_against_all_three_anchors"]
        and checks["pooled_cost_ci_excludes_zero"]
        and checks["pooled_v_rel_ci_excludes_zero"]
    )
    bounded_mechanism = (
        checks["candidate_feasibility_one"]
        and checks["cost_mean_better_against_all_three_anchors"]
        and checks["v_rel_mean_better_against_all_three_anchors"]
    )
    claim_class = (
        "stable_joint_primary_candidate"
        if stable_primary
        else "bounded_mechanism_candidate"
        if bounded_mechanism
        else "unsupported"
    )
    m74 = load(args.m74)
    report = {
        "version": "chip-m75-seed-diverse-decision-v1",
        "status": (
            "pass_stable_primary"
            if stable_primary
            else "pass_bounded_mechanism"
            if bounded_mechanism
            else "fail"
        ),
        "claim_class": claim_class,
        "candidate_report": str(args.candidate),
        "m74_report": str(args.m74),
        "policy": candidate_report["policy"],
        "cases": len(candidate),
        "seeds": seeds,
        "pooled_case_cluster": case_cluster,
        "pooled_naive_pairs": pooled,
        "checks": checks,
        "cost_mean_anchor_wins": cost_mean_wins,
        "v_rel_mean_anchor_wins": v_rel_mean_wins,
        "m74_claim_class": m74["claim_class"],
        "selection_counts": candidate_report["selection_counts"],
        "budget": candidate_report["budget"],
        "claim_boundary": (
            "The candidate is fixed across anchors, but uses three M48 latent "
            "seeds plus deterministic M61B/M65. This is not same-inference, "
            "not training-seed replication and not external PPA evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "claim_class": claim_class,
                "cost_anchor_wins": cost_mean_wins,
                "v_rel_anchor_wins": v_rel_mean_wins,
                "pooled_cost": pooled["paired"]["cost"],
                "pooled_case_cluster_cost": case_cluster["cost"],
                "pooled_case_cluster_v_rel": case_cluster["v_rel"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

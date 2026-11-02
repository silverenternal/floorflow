#!/usr/bin/env python3
"""Aggregate the v45 three-seed matched comparison against released Trinity d192."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import clustered_paired_delta, paired_delta

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 20, 21)
DEVICES = {19: 1, 20: 0, 21: 1}
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-template",
        default="outputs/reports/chip_v45_candidate_official_1600_seed{seed}_group2.json",
    )
    parser.add_argument(
        "--anchor-template",
        default=(
            "outputs/reports/chip_v45_trinity_d192_official_1600_"
            "seed{seed}_cuda{device}.json"
        ),
    )
    parser.add_argument(
        "--candidate-paths",
        nargs=3,
        type=Path,
        help="Optional explicit candidate paths in seed order 19, 20, 21.",
    )
    parser.add_argument(
        "--anchor-paths",
        nargs=3,
        type=Path,
        help="Optional explicit anchor paths in seed order 19, 20, 21.",
    )
    parser.add_argument(
        "--scope",
        default=(
            "Official-100, NFE4, group=2 boundary=2, 1600 linear refiner steps, "
            "candidate training seeds 19-21 with fixed inference seed 19, same GPU per pair"
        ),
    )
    parser.add_argument("--expected-candidate-projector-iterations", type=int, default=8)
    parser.add_argument("--expected-inference-seed", type=int, default=19)
    parser.add_argument(
        "--devices",
        nargs=3,
        type=int,
        help="Optional GPU indices in seed order 19, 20, 21.",
    )
    parser.add_argument(
        "--claim-boundary",
        default=(
            "A pass supports matched-step comparison against released Trinity d192. "
            "It does not establish external routed-PPA improvement."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_matched_compute_three_seed.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def means(rows: list[dict]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }


def main() -> None:
    args = parse_args()
    devices = tuple(args.devices or (DEVICES[seed] for seed in SEEDS))
    candidate_rows = {}
    candidate_reports = {}
    anchor_rows = {}
    anchor_reports = {}
    for position, seed in enumerate(SEEDS):
        candidate_path = (
            args.candidate_paths[position]
            if args.candidate_paths
            else ROOT / args.candidate_template.format(seed=seed)
        )
        anchor_path = (
            args.anchor_paths[position]
            if args.anchor_paths
            else ROOT
            / args.anchor_template.format(
                seed=seed,
                device=devices[position],
            )
        )
        candidate_path = candidate_path.resolve()
        anchor_path = anchor_path.resolve()
        candidate_report = load(candidate_path)
        anchor_report = load(anchor_path)
        candidate_rows[seed] = load(
            candidate_path.with_name(f"{candidate_path.stem}_cases.json")
        )["nfe4_refine1600"]
        anchor_rows[seed] = anchor_report["rows"]
        candidate_reports[seed] = candidate_report
        anchor_reports[seed] = anchor_report

    seed_results = {}
    for seed in SEEDS:
        candidate_mean = means(candidate_rows[seed])
        anchor_mean = means(anchor_rows[seed])
        seed_results[str(seed)] = {
            "device": f"cuda:{devices[SEEDS.index(seed)]}",
            "candidate": candidate_mean,
            "anchor": anchor_mean,
            "gap": {
                "cost_relative": (
                    candidate_mean["legalized_hard_cost"]
                    / anchor_mean["legalized_hard_cost"]
                    - 1.0
                ),
                "v_rel_relative": (
                    candidate_mean["legalized_v_rel"]
                    / anchor_mean["legalized_v_rel"]
                    - 1.0
                ),
            },
            "beats_cost": (
                candidate_mean["legalized_hard_cost"]
                < anchor_mean["legalized_hard_cost"]
            ),
            "beats_v_rel": (
                candidate_mean["legalized_v_rel"]
                < anchor_mean["legalized_v_rel"]
            ),
            "beats_both": (
                candidate_mean["legalized_hard_cost"]
                < anchor_mean["legalized_hard_cost"]
                and candidate_mean["legalized_v_rel"]
                < anchor_mean["legalized_v_rel"]
            ),
            "feasible_fraction": float(
                candidate_reports[seed]["results"]["nfe4_refine1600"][
                    "legalized_feasible_fraction"
                ]
            ),
            "paired_cost": paired_delta(
                candidate_rows[seed],
                anchor_rows[seed],
                "legalized_hard_cost",
                iterations=10000,
                seed=20271000 + seed,
                lower_is_better=True,
            ),
            "paired_v_rel": paired_delta(
                candidate_rows[seed],
                anchor_rows[seed],
                "legalized_v_rel",
                iterations=10000,
                seed=20272000 + seed,
                lower_is_better=True,
            ),
        }

    all_candidate = [row for seed in SEEDS for row in candidate_rows[seed]]
    all_anchor = [row for seed in SEEDS for row in anchor_rows[seed]]
    pooled_cost = clustered_paired_delta(
        all_candidate,
        all_anchor,
        "legalized_hard_cost",
        iterations=10000,
        seed=20273001,
    )
    pooled_v_rel = clustered_paired_delta(
        all_candidate,
        all_anchor,
        "legalized_v_rel",
        iterations=10000,
        seed=20273002,
    )
    checks = {
        "common_inference_seed": all(
            candidate_reports[seed].get("seed") == args.expected_inference_seed
            and anchor_reports[seed].get("protocol", {}).get("seed")
            == args.expected_inference_seed
            for seed in SEEDS
        ),
        "candidate_protocol_consistent": all(
            candidate_reports[seed].get("projector_iterations")
            == args.expected_candidate_projector_iterations
            for seed in SEEDS
        ),
        "all_seeds_beat_cost": all(
            row["beats_cost"] for row in seed_results.values()
        ),
        "all_seeds_beat_v_rel": all(
            row["beats_v_rel"] for row in seed_results.values()
        ),
        "all_seeds_beat_both": all(
            row["beats_both"] for row in seed_results.values()
        ),
        "all_seeds_feasible": all(
            row["feasible_fraction"] == 1.0
            for row in seed_results.values()
        ),
        "pooled_cost_cluster_ci_excludes_zero": pooled_cost["ci_high"] < 0.0,
        "pooled_v_rel_cluster_ci_excludes_zero": pooled_v_rel["ci_high"] < 0.0,
    }
    report = {
        "version": "chip-v45-matched-compute-three-seed-v1",
        "status": "pass_three_seed_matched_compute" if all(checks.values()) else "rejected",
        "scope": args.scope,
        "candidate_train_seeds": list(SEEDS),
        "seeds": seed_results,
        "pooled": {
            "cost": pooled_cost,
            "v_rel": pooled_v_rel,
        },
        "pooled_rowwise_diagnostic": {
            "cost": paired_delta(
                all_candidate,
                all_anchor,
                "legalized_hard_cost",
                iterations=10000,
                seed=20273001,
                lower_is_better=True,
            ),
            "v_rel": paired_delta(
                all_candidate,
                all_anchor,
                "legalized_v_rel",
                iterations=10000,
                seed=20273002,
                lower_is_better=True,
            ),
        },
        "checks": checks,
        "candidate_projector_iterations": {
            str(seed): candidate_reports[seed].get("projector_iterations")
            for seed in SEEDS
        },
        "inference_seed": {
            "candidate": {
                str(seed): candidate_reports[seed].get("seed")
                for seed in SEEDS
            },
            "anchor": {
                str(seed): anchor_reports[seed].get("protocol", {}).get("seed")
                for seed in SEEDS
            },
            "expected": args.expected_inference_seed,
        },
        "claim_boundary": args.claim_boundary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()

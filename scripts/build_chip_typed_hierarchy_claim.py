#!/usr/bin/env python3
"""Freeze the bounded typed-hierarchy chip claim from measured evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    raw = read("outputs/reports/chip_typed_hierarchy_three_seed_v2.json")
    ema = read("outputs/reports/chip_typed_hierarchy_three_seed_v3_ema.json")
    ablation = read("outputs/reports/chip_typed_hierarchy_ablation_v3.json")
    promoted = read(
        "outputs/reports/chip_typed_hierarchy_promoted_controls_v1.json"
    )
    bbox_5k = read(
        "outputs/reports/chip_typed_hierarchy_bbox_repair_5k_v1.json"
    )
    bbox_20k = read(
        "outputs/reports/chip_typed_hierarchy_bbox_repair_20k_v1.json"
    )
    screen = read(
        "outputs/reports/chip_typed_hierarchy_diagnostics_v2.json"
    )
    integration = read(
        "outputs/reports/chip_hierarchical_integration_cpu_v1.json"
    )
    trinity = read(
        "outputs/reports/chip_trinity_d192_component_gap_v1.json"
    )
    candidate = ema["seed_mean"]["candidate"]
    control = ema["seed_mean"]["baseline"]
    trinity_mean = trinity["main_comparison"]["trinity"]
    relative = ema["seed_mean"]["relative_delta"]
    checks = {
        "ema_three_seed_pass": ema["status"] == "pass",
        "raw_negative_retained": raw["status"] == "mixed_or_rejected",
        "all_seed_legality_pass": ema["checks"][
            "all_seeds_legalized_feasible"
        ],
        "mechanism_ablation_pass": (
            ablation["status"] == "pass_with_hierarchy_boundary"
        ),
        "promoted_controls_pass": (
            promoted["status"] == "pass_with_reported_tradeoffs"
        ),
        "bbox_5k_pass": bbox_5k["status"] == "pass",
        "bbox_20k_fixed_weight_rejected": (
            bbox_20k["status"] == "reject_bbox_repair"
        ),
        "integration_cpu_pass": integration["status"] == "pass",
        "external_anchor_available": trinity_mean["mean_cost"] > 0.0,
        "nfe_budget_json_present": (
            ROOT
            / "outputs/chip_hierarchy/v10_three_seed/"
            "reproducible_v1/candidate_seed19_ema_nfe.json"
        ).is_file(),
        "final_figure_present": all(
            (
                ROOT
                / f"outputs/figures/chip_typed_hierarchy_final_pareto_v1.{suffix}"
            ).is_file()
            for suffix in ("pdf", "svg")
        ),
    }
    report = {
        "status": (
            "bounded_positive_ready_for_manuscript"
            if all(checks.values())
            else "not_ready"
        ),
        "scope": "pure chips CFM + exact-area decoder, FloorSet official-100",
        "main_claim": (
            "Under the same CFM, exact-area decoder and EMA evaluation, "
            "augmenting graph-conditioned per-block flow with typed "
            "structural/port conditioning, hierarchical trajectories, "
            "annealed L0 bbox supervision and per-step projection improves "
            "the three-seed mean legalized cost, V_rel, HPWL, area/outline, "
            "b2b and grouping. The effect is heterogeneous across seeds and "
            "is not claimed per case."
        ),
        "supported": {
            "seed_mean_cost_relative_delta": relative["legalized_hard_cost"],
            "seed_mean_v_rel_relative_delta": relative["legalized_v_rel"],
            "seed_mean_hpwl_relative_delta": relative["legalized_hpwl_gap"],
            "seed_mean_area_relative_delta": relative["legalized_area_gap"],
            "seed_mean_b2b_relative_delta": relative[
                "legalized_b2b_hpwl_gap"
            ],
            "seed_mean_p2b_relative_delta": relative[
                "legalized_p2b_hpwl_gap"
            ],
            "seed_mean_grouping_relative_delta": relative["legalized_grouping"],
            "all_seed_legalized_feasible_fraction": 1.0,
            "typed_attention_beats_dense": ablation["checks"][
                "full_beats_dense_grouping"
            ]
            and ablation["checks"]["full_beats_dense_b2b"],
            "netlist_removal_degrades_cost": ablation["checks"][
                "netlist_removal_degrades_cost"
            ],
            "projector_improves_cost": ablation["checks"][
                "projector_improves_cost"
            ],
            "nfe_budget": (
                "Candidate NFE4 matches or beats graph-token NFE8 on both "
                "cost and V_rel in the seed-19 EMA budget curve."
            ),
        },
        "primary_table": {
            "candidate_ema_seed_mean": candidate,
            "graph_token_control_ema_seed_mean": control,
            "relative_delta": relative,
            "per_seed": {
                seed: {
                    "candidate": values["candidate"],
                    "control": values["baseline"],
                    "relative_delta": values["relative_delta"],
                }
                for seed, values in ema["seeds"].items()
            },
        },
        "external_anchor": {
            "trinity_d192_reproducible": trinity_mean,
            "candidate_gap": {
                "cost_relative": (
                    candidate["legalized_hard_cost"]
                    / trinity_mean["mean_cost"]
                    - 1.0
                ),
                "v_rel_relative": (
                    candidate["legalized_v_rel"]
                    / trinity_mean["mean_v_rel"]
                    - 1.0
                ),
            },
            "claim": "Trinity remains stronger; no FloorSet SOTA claim.",
        },
        "statistical_boundary": {
            "pooled_paired": ema["pooled_paired"],
            "warning": (
                "Pooled case intervals for the main metrics cross zero; the "
                "result is a seed-mean effect with heterogeneity, not a "
                "uniform per-case improvement."
            ),
            "raw_evaluation_rejected": raw["status"],
        },
        "boundaries": [
            "No FloorSet SOTA claim.",
            "No all-seed or all-case win claim.",
            "MIB has no effective separation.",
            "No opening, port-direction, halo or congestion claim.",
            "Trinity d192 remains an external anchor.",
            "No single-component attribution of the Trinity gap.",
            "No claim that the hierarchy improves grouping at the promoted controls; no-hierarchy has lower grouping error.",
            "No claim that fixed-weight L0 bbox supervision works at 20k; it was rejected.",
        ],
        "negative_results_retained": {
            "raw_three_seed": raw,
            "fixed_bbox_20k": bbox_20k,
            "5k_screen": {
                "status": screen["status"],
                "checks": screen["checks"],
            },
        },
        "readiness": {
            "experiments_complete": all(checks.values()),
            "paper_written": False,
            "next_external_step": "Manuscript and rebuttal writing only.",
        },
        "checks": checks,
    }
    output = ROOT / "outputs/reports/chip_pure_cvpr_claim_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()

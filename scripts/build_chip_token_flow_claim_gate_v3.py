#!/usr/bin/env python3
"""Freeze claim v3 with mechanism, tail, budget, and external-anchor evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_claim_gate_v3.json",
    )
    return parser.parse_args()


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def structural_table(mechanism: dict) -> list[dict]:
    rows = []
    labels = {
        "legalized_b2b_hpwl_gap": "b2b connectivity HPWL",
        "legalized_area_gap": "Area/outline gap",
        "legalized_grouping_vrel": "Grouping V_rel",
        "legalized_rho_mae": "Rho/aspect MAE",
        "legalized_boundary_vrel": "Boundary V_rel",
        "legalized_p2b_hpwl_gap": "p2b/port HPWL",
        "legalized_mib_vrel": "MIB V_rel",
    }
    for metric, label in labels.items():
        value = mechanism["refined_pooled"][metric]
        rows.append(
            {
                "metric": label,
                "relative_improvement": value["relative_improvement"],
                "ci_low": value["ci_low"],
                "ci_high": value["ci_high"],
                "ci_excludes_zero": (
                    value["ci_low"] < 0.0 and value["ci_high"] < 0.0
                ),
            }
        )
    return rows


def tail_table(mechanism: dict) -> list[dict]:
    cost = mechanism["tail_quantiles"]["legalized_hard_cost"]
    v_rel = mechanism["tail_quantiles"]["legalized_v_rel"]
    return [
        {
            "quantile": quantile,
            "cost_improvement": cost[quantile]["relative_improvement"],
            "v_rel_improvement": v_rel[quantile]["relative_improvement"],
        }
        for quantile in ("0.5", "0.75", "0.9", "0.95")
    ]


def budget_table(mechanism: dict, budget: dict, pareto: dict) -> dict:
    return {
        "nfe": {
            "candidate_nfe": 1,
            "baseline_nfe": 8,
            "candidate": pareto["points"]["nfe"]["1"],
            "baseline": {
                "cost": pareto["points"]["nfe"]["8"]["baseline_cost"],
                "v_rel": pareto["points"]["nfe"]["8"]["baseline_v_rel"],
            },
            "saved_nfe": budget["nfe"]["saved_nfe_vs_8"],
        },
        "refiner": {
            "candidate_steps": 100,
            "baseline_steps": 400,
            "candidate": pareto["points"]["refiner"]["100"],
            "baseline": {
                "cost": pareto["points"]["refiner"]["400"]["baseline_cost"],
                "v_rel": pareto["points"]["refiner"]["400"]["baseline_v_rel"],
            },
            "saved_steps": budget["refiner"]["saved_steps"],
            "fifty_step_boundary": (
                "Candidate 50-step cost is better, but V_rel remains worse "
                "than baseline 400-step quality."
            ),
        },
        "primary_endpoint": "NFE4 + 400 refiner steps",
    }


def main() -> None:
    args = parse_args()
    primary = read("outputs/reports/chip_trinity_token_three_seed_v1.json")
    mechanism = read(
        "outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json"
    )
    budget = read("outputs/reports/chip_token_flow_budget_curve_v1.json")
    pareto = read("outputs/reports/chip_token_flow_pareto_v1.json")
    gap = read("outputs/reports/chip_trinity_d192_component_gap_v1.json")
    checks = {
        "primary_three_seed_pass": primary["status"] == "pass",
        "structural_mechanism": mechanism["checks"][
            "structural_component_supported"
        ],
        "tail_improvement": mechanism["checks"]["tail_improvement_supported"],
        "nfe_budget": budget["checks"]["nfe_budget_reduced"],
        "refiner_budget": budget["checks"]["refiner_budget_reduced"],
        "pareto_assets_complete": all(
            (ROOT / relative).is_file()
            for relative in (
                "outputs/figures/chip_token_flow_pareto_v1.svg",
                "outputs/figures/chip_token_flow_pareto_v1.pdf",
            )
        ),
        "trinity_anchor_audited": gap["status"]
        == "pass_with_legacy_reproducibility_warning",
    }
    selected = all(checks.values())
    candidate = primary["summary"]["refined"]["absolute"]["candidate"]
    main_comparison = gap["main_comparison"]
    external_table = [
        {
            "comparison": "Trinity d192 (unseeded legacy)",
            "cost": main_comparison["legacy_reference"]["trinity"]["cost"],
            "v_rel": main_comparison["legacy_reference"]["trinity"]["v_rel"],
            "role": "appendix_only",
        },
        {
            "comparison": "Trinity d192 (3-seed reproducible)",
            "cost": main_comparison["trinity"]["mean_cost"],
            "v_rel": main_comparison["trinity"]["mean_v_rel"],
            "role": "primary",
        },
        {
            "comparison": "Candidate",
            "cost": candidate["legalized_hard_cost"],
            "v_rel": candidate["legalized_v_rel"],
            "role": "candidate",
        },
        {
            "comparison": "Gap vs legacy",
            "cost_relative": main_comparison["legacy_reference"][
                "candidate_relative_gap"
            ]["cost_relative"],
            "v_rel_relative": main_comparison["legacy_reference"][
                "candidate_relative_gap"
            ]["v_rel_relative"],
            "role": "appendix_only",
        },
        {
            "comparison": "Gap vs reproducible",
            "cost_relative": main_comparison["candidate_relative_gap"][
                "cost_relative"
            ],
            "v_rel_relative": main_comparison["candidate_relative_gap"][
                "v_rel_relative"
            ],
            "role": "reported_primary_gap",
        },
    ]
    report = {
        "status": "selected" if selected else "blocked",
        "checks": checks,
        "main_claim": (
            "在相同 CFM 和 exact-area decoder 下，将 graph conditioning 从 "
            "invariant global slot 改为 per-block trajectory，主要改善 b2b "
            "connectivity、area/outline、grouping 和 cost 长尾；在 pooled "
            "diagnostics 中总 V_rel 改善 -11.89% 且 CI 排除零；同时 candidate "
            "能以更少 NFE 或 refiner steps 达到与 baseline 相当的质量。"
        ),
        "main_tables": {
            "structural_relations": structural_table(mechanism),
            "quantiles": tail_table(mechanism),
            "equal_quality_budget": budget_table(mechanism, budget, pareto),
            "external_anchor": external_table,
        },
        "reported_primary_gap": main_comparison["candidate_relative_gap"],
        "mechanism_explanation": (
            "The strongest reproducible signals concern module-to-module "
            "structure: b2b connectivity, area/outline, grouping, and aspect. "
            "Per-block graph conditioning therefore improves local structural "
            "generation rather than only reducing a global statistic."
        ),
        "complexity_explanation": (
            "Gains concentrate in middle complexity and low-b2b/pin strata; the "
            "highest-b2b and highest-pin strata show little gain. Under very tight "
            "topology or high ambiguity, changing only the conditioning injection "
            "path is insufficient."
        ),
        "mib_explanation": (
            "MIB V_rel is zero for both arms on official-100 and is not a positive result."
        ),
        "external_anchor": {
            "primary": "three_seed_reproducible",
            "main_table": external_table,
            "legacy_no_seed": {
                "trinity": gap["legacy_anchor"]["summary"],
                "candidate_relative_gap": gap["legacy_anchor"][
                    "candidate_relative_gap"
                ],
            },
            "three_seed_replication": {
                "trinity": gap["reproducible_replication"]["seed_mean"],
                "range": gap["reproducible_replication"]["seed_ranges"],
                "candidate_relative_gap": gap["reproducible_replication"][
                    "candidate_relative_gap"
                ],
            },
            "component_gap": gap["reproducible_replication"][
                "paired_component_gap"
            ],
            "warning": (
                "Keep the legacy +5.76%/+30.45% boundary as predeclared, but "
                "report that three fixed seeds did not reproduce that optimistic "
                "one-sample anchor."
            ),
        },
        "required_boundaries": [
            "FloorSet SOTA is not claimed.",
            "No per-case all-win claim.",
            "No MIB improvement claim.",
            "No opening-specific claim; FloorSet has no opening metric.",
            "No benefit claim for highest-b2b or highest-pin strata.",
            "Graph2Plan and FreeScene are related-work context only.",
            "HouseMind has no verified primary record and no numeric comparison.",
            "Budget curves are diagnostics; NFE4 + 400 remains primary.",
            (
                "Component gaps describe structural differences and are not "
                "attributed to any single factor among exact-area decoder, "
                "training data, or sampling strategy."
            ),
        ],
        "related_work_positioning": {
            "trinity": (
                "Trinity d192 is the legacy anchor under the current protocol; its "
                "full training and sampling configuration is not completely public. "
                "The reported external gap uses the reproducible three-seed anchor, "
                "with no cross-protocol SOTA claim."
            ),
            "graph2plan_freescene": (
                "Graph2Plan and FreeScene address adjacency-graph layout generation "
                "and free-prompt 3D scene synthesis respectively; their tasks and "
                "metrics are not directly comparable to CFM placement."
            ),
            "housemind": (
                "No verified primary record was found; no HouseMind numbers are used."
            ),
        },
        "evidence": {
            "primary": "outputs/reports/chip_trinity_token_three_seed_v1.json",
            "mechanism": "outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json",
            "budget": "outputs/reports/chip_token_flow_budget_curve_v1.json",
            "pareto": "outputs/reports/chip_token_flow_pareto_v1.json",
            "trinity_gap": "outputs/reports/chip_trinity_d192_component_gap_v1.json",
        },
        "candidate_refined": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

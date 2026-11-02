#!/usr/bin/env python3
"""Build the mechanism- and budget-aware graph-token flow claim gate."""

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
        default=ROOT / "outputs/reports/chip_token_flow_claim_gate_v2.json",
    )
    return parser.parse_args()


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    primary = read("outputs/reports/chip_trinity_token_three_seed_v1.json")
    mechanism = read(
        "outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json"
    )
    budget = read("outputs/reports/chip_token_flow_budget_curve_v1.json")
    external = read("outputs/reports/trinity_d192_matched_protocol_v2.json")
    candidate = primary["summary"]["refined"]["absolute"]["candidate"]
    baseline = primary["summary"]["refined"]["absolute"]["baseline"]
    checks = {
        "three_seed_primary_pass": primary["status"] == "pass",
        "structural_mechanism_supported": mechanism["checks"][
            "structural_component_supported"
        ],
        "tail_improvement_supported": mechanism["checks"][
            "tail_improvement_supported"
        ],
        "equal_quality_nfe_supported": budget["checks"]["nfe_budget_reduced"],
        "equal_quality_refiner_supported": budget["checks"][
            "refiner_budget_reduced"
        ],
        "external_same_protocol_reported": external["mean_cost"] > 0.0,
    }
    cost_gap = candidate["legalized_hard_cost"] / external["mean_cost"] - 1.0
    v_rel_gap = candidate["legalized_v_rel"] / external["mean_v_rel"] - 1.0
    selected = all(checks.values())
    report = {
        "status": "selected" if selected else "blocked",
        "checks": checks,
        "claim": {
            "id": "B_graph_token_flow_mechanism_budget",
            "statement": (
                "Under the same CFM objective and exact-area decoder, moving graph "
                "conditioning from an invariant global slot to per-block flow "
                "trajectories improves structural geometry and reduces cost-tail "
                "failures. The matched model also reaches declared baseline quality "
                "with fewer NFE and refiner steps."
            ),
            "mechanism": (
                "The strongest matched components are b2b connectivity HPWL, "
                "area/outline gap, rho/aspect error, and cluster grouping; these "
                "support a structural-relation explanation."
            ),
            "budget": {
                "candidate_minimum_nfe_for_baseline_nfe8_quality": budget["nfe"][
                    "candidate_minimum_nfe"
                ],
                "saved_nfe": budget["nfe"]["saved_nfe_vs_8"],
                "candidate_minimum_refiner_steps": budget["refiner"][
                    "candidate_minimum_steps"
                ],
                "saved_refiner_steps": budget["refiner"]["saved_steps"],
            },
            "primary_effect": {
                "refined_seed_mean_relative": primary["summary"]["refined"][
                    "seed_mean_relative_delta"
                ],
                "raw_seed_mean_relative": primary["summary"]["raw"][
                    "seed_mean_relative_delta"
                ],
            },
            "component_effect": {
                metric: mechanism["refined_pooled"][metric][
                    "relative_improvement"
                ]
                for metric in (
                    "legalized_b2b_hpwl_gap",
                    "legalized_p2b_hpwl_gap",
                    "legalized_area_gap",
                    "legalized_grouping_vrel",
                    "legalized_mib_vrel",
                    "legalized_boundary_vrel",
                    "legalized_rho_mae",
                )
            },
            "tail_effect": mechanism["tail_quantiles"],
            "external_table": {
                "graph_global_slot": baseline,
                "graph_token_flow": candidate,
                "trinity_d192_same_protocol": {
                    "legalized_hard_cost": external["mean_cost"],
                    "legalized_v_rel": external["mean_v_rel"],
                },
                "token_gap_to_trinity": {
                    "cost_relative": cost_gap,
                    "v_rel_relative": v_rel_gap,
                },
            },
            "limitations": [
                "The model remains 5.76% above Trinity on hard cost and 30.45% above it on V_rel.",
                "Complexity gains are not monotonic; high-b2b and high-pin strata show little gain.",
                "MIB shape violation is zero for both arms on official-100.",
                "Boundary and p2b mechanisms have large mean gains but confidence intervals that cross zero.",
                "FloorSet has no floorplan-opening metric; boundary and p2b are not relabeled as openings.",
                "Primary per-case intervals still cross zero on seeds 19 and 21.",
                "The refiner budget curve is seed 19 only; the primary endpoint remains 400 steps.",
            ],
            "not_supported": [
                "FloorSet state of the art",
                "Superiority over same-protocol Trinity d192",
                "Universal improvement on every complexity stratum or every case",
                "An opening-specific improvement",
                "MIB-shape improvement on the official endpoint",
            ],
        },
        "external_context": {
            "numeric_same_protocol": [
                {
                    "name": "Trinity d192",
                    "url": "https://arxiv.org/abs/2610.04957",
                    "metric_comparable": True,
                }
            ],
            "qualitative_only": [
                {
                    "name": "Graph2Plan",
                    "url": "https://arxiv.org/abs/2004.13204",
                    "reason": "RPLAN floorplan retrieval/generation metrics are not FloorSet cost/V_rel.",
                },
                {
                    "name": "FreeScene",
                    "url": "https://arxiv.org/abs/2506.02781",
                    "reason": "3D scene synthesis metrics are not comparable to chip floorplanning.",
                },
            ],
            "unverified": [
                {
                    "name": "HouseMind",
                    "reason": "No verifiable primary arXiv or Crossref record was found.",
                }
            ],
        },
        "evidence": {
            "primary": "outputs/reports/chip_trinity_token_three_seed_v1.json",
            "mechanism": (
                "outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json"
            ),
            "budget": "outputs/reports/chip_token_flow_budget_curve_v1.json",
            "external": "outputs/reports/trinity_d192_matched_protocol_v2.json",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

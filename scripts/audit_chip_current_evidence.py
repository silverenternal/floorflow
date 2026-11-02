#!/usr/bin/env python3
"""Audit the current chip evidence chain without promoting incomplete claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--todo",
        type=Path,
        default=ROOT / "chipstodo.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_current_evidence_gate_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def python_line_sizes(root: Path) -> dict[str, int]:
    sizes = {}
    for directory in (root / "src", root / "scripts", root / "tests"):
        for path in directory.rglob("*.py"):
            sizes[str(path.relative_to(root))] = len(path.read_text().splitlines())
    return sizes


def build_gate(root: Path, todo: dict) -> dict:
    reports = {
        "matched": load(root / "outputs/reports/chip_v45_matched_compute_three_seed.json"),
        "matched_components": load(
            root / "outputs/reports/chip_v45_matched_component_gap_v1.json"
        ),
        "matched_highblock": load(
            root / "outputs/reports/chip_sota_official_highblock8_pair_refine1600_v1.json"
        ),
        "external_ppa": load(
            root / "outputs/reports/chip_external_ppa_holdout_v44_all.json"
        ),
        "adaptive": load(
            root / "outputs/reports/chip_sota_adaptive_signal_calibration_dev32_v1.json"
        ),
        "u3": load(
            root
            / "outputs/reports/chip_v46_bookshelf_outline_fallback_full22_16draw_coverage_v1.json"
        ),
        "u3_seed": load(
            root / "outputs/reports/chip_outline_seed_stability_v1.json"
        ),
        "u3_failure": load(
            root / "outputs/reports/chip_outline_failure_modes_v1.json"
        ),
        "u3_fallback_effect": load(
            root / "outputs/reports/chip_outline_fallback_effect_16draw_v1.json"
        ),
        "orfs_supply": load(
            root / "outputs/reports/chip_orfs_macro_candidate_scan_v1.json"
        ),
        "chipbench_supply": load(
            root / "outputs/reports/chip_chipbench_holdout_manifest_v1.json"
        ),
        "observed_external": load(
            root / "outputs/reports/chip_external_holdout_v45_manifest_v2.json"
        ),
        "capacity_b21": load(
            root / "outputs/reports/chip_capacity_budget_b21_v1.json"
        ),
        "capacity_b40": load(
            root / "outputs/reports/chip_capacity_budget_v1.json"
        ),
        "capacity_b120": load(
            root / "outputs/reports/chip_capacity_budget_b120_v1.json"
        ),
        "training_comparison": load(
            root / "outputs/reports/chip_sota_training_old_vs_new20k_dev32_cuda1.json"
        ),
        "training_subset_control": load(
            root / "outputs/reports/chip_sota_training_subset_control_v2.json"
        ),
        "training_subset_comparison": load(
            root / "outputs/reports/chip_sota_training_old_vs_subset_control_v2_cuda1.json"
        ),
        "training_full50k": load(
            root / "outputs/reports/chip_sota_training_full50k_exposure_v1.json"
        ),
        "u1_screen": load(
            root / "outputs/reports/chip_u1_dynamic_graph_screen_v1.json"
        ),
        "m10_smoke": load(
            root
            / "outputs/smoke/chip_m10_projection_consistent_flow_smoke_v1.json"
        ),
        "m10_nfe4_smoke": load(
            root
            / "outputs/smoke/chip_m10_projection_consistent_nfe4_smoke_v1.json"
        ),
        "m10_fallback_quick": load(
            root
            / "outputs/reports/chip_m10_projection_consistent_flow_dev32_fallback_cpu_v1.json"
        ),
        "m10_selection": load(
            root / "outputs/reports/chip_m10_quick_selection_v1.json"
        ),
        "m10_confirmation": load(
            root
            / "outputs/reports/chip_m10_projection_consistent_flow_final_only_confirmation_fallback_cpu_v1.json"
        ),
        "m6_repair": load(
            root / "outputs/reports/chip_v46_response_repair_queue.json"
        ),
        "m6_decision": load(
            root / "outputs/reports/chip_v46_analytical_search_decision_v1.json"
        ),
        "m48_portfolio": load(
            root / "outputs/reports/chip_m48_portfolio_three_seed_v1.json"
        ),
    }
    reuse_missing = []
    for task in todo["tasks"]:
        for reference in task.get("reuse", []):
            if reference and not (root / reference).exists():
                reuse_missing.append({"task": task["id"], "path": reference})
    task_ids = [task["id"] for task in todo["tasks"]]
    line_sizes = python_line_sizes(root)
    oversized = {
        path: lines
        for path, lines in line_sizes.items()
        if lines >= 600
    }
    observed_ids = {
        case["design_key"]
        for case in reports["observed_external"]["cases"]
    }
    static_ids = {
        case["id"]
        for case in reports["orfs_supply"]["candidates"]
        if case["platform_available"] and case["macro_master_count"] <= 64
    }
    static_stdcell_expansions = {
        case["id"]
        for case in reports["orfs_supply"]["candidates"]
        if case["platform_available"] and case["macro_master_count"] > 64
    }
    fresh_static_ids = sorted(static_ids - observed_ids)
    capacity_rows = {
        block_count: {
            row["width"]: row["training_forward_backward_flops"]
            for row in report["rows"]
        }
        for block_count, report in (
            (21, reports["capacity_b21"]),
            (40, reports["capacity_b40"]),
            (120, reports["capacity_b120"]),
        )
    }
    integrity_checks = {
        "todo_version_present": bool(todo.get("version")),
        "task_ids_unique": len(task_ids) == len(set(task_ids)),
        "active_reuse_paths_exist": not reuse_missing,
        "python_files_under_600_lines": not oversized,
        "required_reports_present": all(
            isinstance(report, dict) and report for report in reports.values()
        ),
    }
    scientific_gates = {
        "matched_compute_candidate_beats_anchor": (
            reports["matched"]["status"] != "rejected"
        ),
        "matched_highblock_pilot_supported": (
            reports["matched_highblock"]["status"] != "pilot_not_supported"
        ),
        "released_anchor_vrel_grouping_all_three_seeds": (
            reports["matched_components"]["summary"]["v_rel"][
                "seed_improvements"
            ]
            == 3
            and reports["matched_components"]["summary"]["grouping"][
                "seed_improvements"
            ]
            == 3
        ),
        "external_routed_ppa_supported": (
            reports["external_ppa"]["status"] != "external_ppa_not_supported"
        ),
        "adaptive_refiner_promoted": not (
            reports["adaptive"]["policy"]["kind"] == "fixed"
            and reports["adaptive"]["policy"]["steps"] == 1600
        ),
        "u3_candidate_coverage_significant": (
            reports["u3"]["feasibility_mcnemar_exact_p"] < 0.05
        ),
        "u3_hpwl_significant": (
            reports["u3"]["feasible_pair_hpwl"]["ci_high"] < 0.0
        ),
        "u3_seed_candidate_coverage_significant": (
            reports["u3_seed"]["feasibility_mcnemar_exact_p"] < 0.05
        ),
        "u3_candidate_hard_advantage": (
            reports["u3_failure"]["variants"]["hard"]["candidate"][
                "fully_feasible"
            ]
            > reports["u3_failure"]["variants"]["hard"]["control"][
                "fully_feasible"
            ]
        ),
        "u3_fallback_hard_gain": (
            reports["u3_fallback_effect"]["variants"]["hard"]["candidate"][
                "with_fallback"
            ]["feasible"]
            > reports["u3_fallback_effect"]["variants"]["hard"]["candidate"][
                "without_fallback"
            ]["feasible"]
            and reports["u3_fallback_effect"]["variants"]["hard"]["control"][
                "with_fallback"
            ]["feasible"]
            > reports["u3_fallback_effect"]["variants"]["hard"]["control"][
                "without_fallback"
            ]["feasible"]
        ),
        "fresh_orfs_supply_sufficient": (
            len(fresh_static_ids) >= 15
        ),
        "fresh_external_supply_sufficient": (
            reports["chipbench_supply"]["status"].startswith(
                "frozen_before_candidate_outcome"
            )
            and reports["chipbench_supply"]["design_count"] >= 15
            and all(reports["chipbench_supply"]["checks"].values())
        ),
        "training_20k_combined_recipe_beats_production_control": (
            reports["training_comparison"]["status"]
            == "candidate_passes_same_device_gate"
        ),
        "training_subset_control_complete": (
            reports["training_subset_control"]["status"].startswith("complete")
        ),
        "training_subset_control_matches_production": (
            reports["training_subset_comparison"]["status"]
            == "candidate_passes_same_device_gate"
        ),
        "training_full50k_exposure_pass": (
            reports["training_full50k"]["status"] == "complete"
        ),
        "u1_quick_mean_pass": (
            bool(reports["u1_screen"]["stages"])
            and reports["u1_screen"]["stages"][0]["decision"]["advance"]
        ),
        "u1_confirmation_complete": (
            len(reports["u1_screen"]["stages"]) > 1
            and bool(reports["u1_screen"]["stages"][1].get("decision"))
        ),
        "u1_confirmation_pass": (
            len(reports["u1_screen"]["stages"]) > 1
            and bool(reports["u1_screen"]["stages"][1]["decision"].get("advance"))
        ),
        "m10_operator_smoke_complete": (
            reports["m10_smoke"]["status"] == "complete"
            and reports["m10_smoke"]["cases"] >= 1
            and len(reports["m10_smoke"]["summaries"]) >= 2
            and reports["m10_nfe4_smoke"]["status"] == "complete"
            and reports["m10_nfe4_smoke"]["cases"] >= 1
        ),
        "m10_frozen_screen_complete": (
            reports["m10_fallback_quick"]["status"] == "complete"
            and reports["m10_fallback_quick"]["cases"] >= 32
            and reports["m10_confirmation"]["status"] == "complete"
            and reports["m10_confirmation"]["cases"] >= 32
        ),
        "m10_screen_pass": (
            reports["m10_selection"]["status"]
            == "final_only_selected_for_confirmation"
            and reports["m10_confirmation"]["any_mean_pass"]
        ),
        "m6_response_repair_complete": (
            not reports["m6_repair"].get("missing_responses", [])
            and all(
                run.get("status") in {"complete", "skipped_already_complete"}
                for run in reports["m6_repair"].get("runs", [])
            )
        ),
        "m6_deployable_selector_supported": (
            reports["m6_decision"]["status"] != "reject_deployable_selector"
        ),
        "m48_seed_stable_absolute_budget_pass": (
            reports["m48_portfolio"]["status"]
            == "seed_stable_absolute_budget_pass"
        ),
    }
    blockers = [
        "matched-compute Trinity comparison remains rejected",
        "the 20k combined training recipe regressed versus the same-device production control",
        "the clean 20k-subset control also regresses versus the production control",
        "the 50k-step full-cache exposure arm regressed cost by 1.71% and V_rel by 20.40% versus the same-device production control",
        "U1 was rejected at the 5k/refiner1600 confirmation because V_rel regressed",
        "M6 response repair is complete, but the two-case pilot rejects a deployable selector because it adds no discriminative-p2b choice and costs 845 seconds",
        "matched component decomposition does not support universal b2b/area gains",
        "external routed-PPA transfer remains unsupported",
        "the 15-design fresh external cohort is frozen, but candidate/control routed outcomes are not yet opened",
        "one external design remains pending its pinned floorplan macro-count preflight",
        "sky130hd/chameleon statically has four macro masters but an estimated six instances because RAM_3Kx32 unrolls three DFFRAM_4K cells; elaboration must resolve the expected count",
        "GRT congestion/DRC/timing/power certification remains incomplete",
        "U3 candidate-vs-control coverage is non-significant, although fallback on/off feasibility is significant",
        "U3 three-seed feasibility probe is positive but non-significant",
        "U3 candidate-vs-control advantage is SOFT-concentrated; fallback itself improves HARD coverage in both arms",
        "M48 is seed-stable on FloorSet under a declared 2x generation budget, but same-inference and external routed-PPA transfer remain unproven",
    ]
    return {
        "version": "chip-current-evidence-gate-v1",
        "todo_version": todo.get("version"),
        "artifact_integrity": {
            "status": "pass" if all(integrity_checks.values()) else "fail",
            "checks": integrity_checks,
            "invalid_reuse_paths": reuse_missing,
            "oversized_python_files": oversized,
            "max_python_lines": max(line_sizes.values()),
        },
        "scientific_gates": scientific_gates,
        "blockers": blockers,
        "capacity_training_flops": capacity_rows,
        "u3_variants": reports["u3"]["variants"],
        "matched_component_summary": reports["matched_components"]["summary"],
        "training_frontier": {
            "comparison": "outputs/reports/chip_sota_training_old_vs_new20k_dev32_cuda1.json",
            "comparison_status": reports["training_comparison"]["status"],
            "cost_relative": reports["training_comparison"]["paired"][
                "legalized_hard_cost"
            ]["relative_delta"],
            "v_rel_relative": reports["training_comparison"]["paired"][
                "legalized_v_rel"
            ]["relative_delta"],
            "subset_control_status": reports["training_subset_control"]["status"],
            "subset_comparison": (
                "outputs/reports/chip_sota_training_old_vs_subset_control_v2_cuda1.json"
            ),
            "subset_cost_relative": reports["training_subset_comparison"][
                "paired"
            ]["legalized_hard_cost"]["relative_delta"],
            "subset_v_rel_relative": reports["training_subset_comparison"][
                "paired"
            ]["legalized_v_rel"]["relative_delta"],
        },
        "u1_screen": {
            "report": "outputs/reports/chip_u1_dynamic_graph_screen_v1.json",
            "status": reports["u1_screen"]["status"],
            "quick_relative": (
                reports["u1_screen"]["stages"][0]["decision"]["relative"]
                if reports["u1_screen"]["stages"]
                else None
            ),
            "confirmation_relative": (
                reports["u1_screen"]["stages"][1]["decision"]["relative"]
                if len(reports["u1_screen"]["stages"]) > 1
                else None
            ),
        },
        "m10_projection_flow": {
            "report": (
                "outputs/smoke/chip_m10_projection_consistent_flow_smoke_v1.json"
            ),
            "stage": reports["m10_smoke"].get("stage"),
            "quick_cases": reports["m10_fallback_quick"]["cases"],
            "confirmation_cases": reports["m10_confirmation"]["cases"],
            "status": reports["m10_selection"]["status"],
            "confirmation_any_mean_pass": reports["m10_confirmation"][
                "any_mean_pass"
            ],
            "nfe4_smoke_cost_delta": reports["m10_nfe4_smoke"][
                "comparisons"
            ]["projection_consistent_nfe4_refine0"]["mean_cost_delta"],
            "scope": "frozen fallback-checkpoint development screen; rejected",
        },
        "m6_response_repair": {
            "report": "outputs/reports/chip_v46_response_repair_queue.json",
            "status": reports["m6_repair"].get("status"),
            "missing_response_count": len(
                reports["m6_repair"].get("missing_responses", [])
            ),
            "runs": len(reports["m6_repair"].get("runs", [])),
            "decision_status": reports["m6_decision"]["status"],
            "decision_cases": reports["m6_decision"]["case_count"],
            "analytical_changes": reports["m6_decision"][
                "analytical_changes_default_p2b_choice"
            ],
            "response_seconds": reports["m6_decision"]["response_seconds"],
        },
        "m48_portfolio": {
            "report": "outputs/reports/chip_m48_portfolio_three_seed_v1.json",
            "status": reports["m48_portfolio"]["status"],
            "budget_class": reports["m48_portfolio"]["budget_class"],
            "pooled_cost": reports["m48_portfolio"]["pooled"]["cost"],
            "pooled_v_rel": reports["m48_portfolio"]["pooled"]["v_rel"],
            "claim_boundary": reports["m48_portfolio"]["claim_boundary"],
        },
        "u3_seed_probe": {
            "candidate_feasible": reports["u3_seed"]["candidate_feasible"],
            "control_feasible": reports["u3_seed"]["control_feasible"],
            "cases": reports["u3_seed"]["cases"],
            "mcnemar_exact_p": reports["u3_seed"]["feasibility_mcnemar_exact_p"],
            "paired_hpwl_relative_mean": reports["u3_seed"][
                "paired_hpwl_relative_mean"
            ],
        },
        "training_full50k": {
            "report": "outputs/reports/chip_sota_training_full50k_exposure_v1.json",
            "status": reports["training_full50k"]["status"],
            "steps": reports["training_full50k"].get("steps", []),
        },
        "u3_failure_modes": {
            "hard": reports["u3_failure"]["variants"]["hard"],
            "soft": reports["u3_failure"]["variants"]["soft"],
        },
        "u3_fallback_effect": reports["u3_fallback_effect"]["overall"],
        "fresh_static_candidates": fresh_static_ids,
        "static_stdcell_expansions": sorted(static_stdcell_expansions),
        "fresh_external_supply": {
            "manifest": "outputs/reports/chip_chipbench_holdout_manifest_v1.json",
            "status": reports["chipbench_supply"]["status"],
            "design_count": reports["chipbench_supply"]["design_count"],
            "family_count": reports["chipbench_supply"]["family_count"],
            "macro_counts_verified": (
                reports["chipbench_supply"]["checks"]["all_macro_counts_verified"]
            ),
            "graph_audits_ready": reports["chipbench_supply"]["checks"][
                "chipbench_graph_audits_pass"
            ],
            "flow_configs_ready": reports["chipbench_supply"]["checks"][
                "chipbench_flow_configs_present"
            ],
            "pending_macro_preflight": [
                row["id"]
                for row in reports["chipbench_supply"]["designs"]
                if row["macro_instance_count"] is None
            ],
            "candidate_outcomes_opened": False,
            "adapter_smoke_cases": reports["chipbench_supply"][
                "adapter_smoke_cases"
            ],
            "adapter_smoke_boundary": reports["chipbench_supply"][
                "adapter_smoke_boundary"
            ],
            "license_note": reports["chipbench_supply"]["licenses"]["decision"],
        },
        "publishable_cvpr_evidence_ready": False,
        "decision": (
            "M48 supports a seed-stable absolute-budget FloorSet mechanism claim, "
            "but not same-inference SOTA or external routed-PPA transfer. Keep "
            "the venue-wide positive claim false until external certification passes."
        ),
    }


def main() -> None:
    args = parse_args()
    gate = build_gate(ROOT, load(args.todo))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()

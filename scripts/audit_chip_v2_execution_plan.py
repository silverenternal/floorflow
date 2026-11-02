#!/usr/bin/env python3
"""Validate the frozen chip v2 stage order, budgets, and safety contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "configs/chip_v2_execution_plan_v1.json",
    )
    parser.add_argument(
        "--architecture",
        type=Path,
        default=ROOT / "configs/chip_architecture_completion_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v2_execution_plan_audit_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_plan(root: Path, plan: dict, architecture: dict) -> dict:
    architecture_stages = {
        stage["id"]: stage for stage in architecture["stages"]
    }
    expected_order = [
        "M50-COMMITTED-HIERARCHY-FLOW",
        "M51-POSITION-SHAPE-FACTORIZATION",
        "M53-REFINER-COMPATIBLE-FLOW",
        "M52-RELATION-SPECIFIC-FUSION",
        "M55-ROUTEABILITY-SPATIAL-STREAM",
        "M57-CONFIDENCE-GATED-COMMIT",
        "M58-SHARED-TRUNK-FACTORIZATION",
        "M59-LEGALIZER-CONSTRAINED-ENDPOINT",
        "M60-INCIDENCE-NORMALIZED-GRAPH",
        "M61A-ROUTE-ONLY",
        "M61B-SOURCE-ONLY",
        "M62-CONSTRAINED-COMPLEMENTARY-FUSION",
        "M64-SOURCE-ONLY-CLEAN",
        "M65-GRAPH-SOURCE-FUSION",
    ]
    expected_architecture_order = [
        "M50-SAME-CALL-COMMIT",
        "M51-FACTORIZED-GEOMETRY",
        "M53-REFINER-COMPATIBLE-FLOW",
        "M52-RELATION-GRAPH",
        "M55-ROUTEABILITY-AND-STABLE-TRAINING",
        "M57-CONFIDENCE-GATED-COMMIT",
        "M58-SHARED-TRUNK-FACTORIZATION",
        "M59-LEGALIZER-CONSTRAINED-ENDPOINT",
        "M60-INCIDENCE-NORMALIZED-GRAPH",
        "M61A-ROUTE-ONLY",
        "M61B-SOURCE-ONLY",
        "M62-CONSTRAINED-FUSION",
        "M64-SOURCE-ONLY-CLEAN",
        "M65-GRAPH-SOURCE-FUSION",
    ]
    required_paths = [
        plan["parent_checkpoint"],
        plan["architecture_config"],
        plan["mechanism_policy"],
        plan["training_subset"],
        plan["validation"]["case_manifest"],
        plan["legalizer_closed_loop"]["train_indices"],
        plan["legalizer_closed_loop"]["response_sidecar"],
    ]
    missing_paths = [
        path for path in required_paths if not (root / path).is_file()
    ]
    stage_mismatches = [
        stage["id"]
        for stage in plan["stages"]
        if stage["architecture_stage"] not in architecture_stages
    ]
    legalizer = plan["legalizer_closed_loop"]
    legalizer_flags = architecture_stages.get(legalizer["architecture_stage"], {}).get(
        "flags", {}
    )
    m53_flags = architecture_stages.get(
        "M53-REFINER-COMPATIBLE-FLOW", {}
    ).get("flags", {})
    m55_flags = architecture_stages.get(
        "M55-ROUTEABILITY-AND-STABLE-TRAINING", {}
    ).get("flags", {})
    checks = {
        "execution_order_is_prescribed": plan["execution_order"] == expected_order,
        "architecture_execution_order_matches": (
            architecture.get("execution_order") == expected_architecture_order
        ),
        "all_referenced_paths_exist": not missing_paths,
        "architecture_stage_ids_match": not stage_mismatches,
        "candidate_control_budget_is_matched": all(
            plan["matched_control"].get(key) is True
            for key in (
                "same_parent",
                "same_training_subset",
                "same_steps_and_effective_batch",
                "same_optimizer_and_ema",
                "same_nfe_refiner_and_legalizer_budget",
                "same_data_order_seed",
                "only_mechanism_flags_may_differ",
            )
        ),
        "checkpoint_resume_state_is_complete": all(
            plan["checkpoint_contract"].get(key) is True
            for key in (
                "atomic_writes",
                "save_model_optimizer_ema",
                "save_torch_numpy_python_rng",
                "save_batch_rng_and_balanced_sampler_state",
                "save_data_order_hash",
                "fail_closed_on_nonfinite",
            )
        ),
        "m53_does_not_apply_legalizer_action_early": (
            "structured_legalizer_apply" not in m53_flags
            and "endpoint_compatibility_weight" in m53_flags
        ),
        "m54_requires_response_labels_before_action": (
            legalizer_flags.get("structured_legalizer_apply") is True
            and legalizer_flags.get("structured_legalizer_response") is True
            and float(legalizer_flags.get("legalizer_response_weight", 0.0)) > 0.0
        ),
        "m55_enables_route_field_and_source_schedule": (
            m55_flags.get("spatial_route_field") is True
            and m55_flags.get("hierarchy_source_mode") == "mixed"
        ),
        "gpu_execution_is_authorized": (
            plan["gpu_execution"] == "authorized_bounded_screens"
        ),
    }
    return {
        "version": "chip-v2-execution-plan-audit-v1",
        "status": "pass_cpu_plan_only" if all(checks.values()) else "fail",
        "checks": checks,
        "missing_paths": missing_paths,
        "stage_mismatches": stage_mismatches,
        "execution_order": plan["execution_order"],
        "legalizer_stage": legalizer["id"],
        "gpu_experiments_run": False,
        "boundary": (
            "This command audits plan and path structure only. GPU screens are "
            "authorized separately and their evidence must be recorded in stage reports."
        ),
    }


def main() -> None:
    args = parse_args()
    report = audit_plan(ROOT, load(args.plan), load(args.architecture))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    if report["status"] != "pass_cpu_plan_only":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the post-fix chip claim registry from frozen CPU-visible evidence."""

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
        default=ROOT / "outputs/reports/chip_current_evidence_gate_postfix_v1.json",
    )
    parser.add_argument(
        "--document",
        type=Path,
        default=ROOT / "docs/chip_evidence_lineage_postfix_v1.md",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _claim(status: str, evidence: str, statement: str, boundary: str) -> dict:
    return {
        "status": status,
        "evidence": evidence,
        "statement": statement,
        "boundary": boundary,
    }


def build_registry(root: Path) -> dict:
    source_paths = {
        "lineage": "outputs/reports/chip_architecture_lineage_audit_v1.json",
        "contract": "outputs/reports/chip_architecture_v2_contract_audit.json",
        "complexity": "outputs/reports/chip_complexity_tail_audit_v1.json",
        "route": "outputs/reports/chip_route_coordinate_audit_v1.json",
        "router": "outputs/reports/chip_portfolio_router_fit_v1.json",
        "invalidated_portfolio": "outputs/reports/chip_m48_portfolio_three_seed_v1.json",
    }
    missing = [path for path in source_paths.values() if not (root / path).is_file()]
    lineage = load(root / source_paths["lineage"])
    paired = lineage["postfix_portfolio_vs_anchor"]["paired"]
    claims = [
        _claim(
            "supported_contract",
            source_paths["contract"],
            "All v2 additions are zero-gated at initialization and the typed "
            "structural and legalizer-response ports are active.",
            "Contract equivalence only; no quality result.",
        ),
        _claim(
            "supported_contract",
            source_paths["lineage"],
            "The canonical M27 lineage has 6,797,217 parameters, 16.06% above "
            "Trinity d192, and starts from the d192 backbone.",
            "Parameter and initialization provenance, not SOTA.",
        ),
        _claim(
            "descriptive_higher_budget",
            source_paths["lineage"],
            "The post-fix three-portfolio join improves pooled cost by 3.78% "
            "and V_rel by 45.62% versus the available NFE1/ref1600 anchor.",
            "Candidate uses NFE2/ref3200/three legalizer calls; no matched-budget claim.",
        ),
        _claim(
            "descriptive_higher_budget",
            source_paths["lineage"],
            "Grouping and boundary improve, while b2b, area and rho remain weaker.",
            "Structural diagnosis under the declared higher-budget portfolio.",
        ),
        _claim(
            "unsupported",
            source_paths["lineage"],
            "No matched-budget Trinity victory or same-inference SOTA is claimed.",
            "An equal-budget anchor remains pending.",
        ),
        _claim(
            "unsupported",
            source_paths["complexity"],
            "Dense high-pin and high-b2b strata remain weak and block a universal "
            "structural-improvement claim.",
            "Descriptive tail evidence only.",
        ),
        _claim(
            "review_only",
            source_paths["route"],
            "Route-field code is sensitive to valid pins, but routed transfer is "
            "not established.",
            "The route coordinate audit still fails the all-pins-inside-bbox check.",
        ),
        _claim(
            "unsupported",
            source_paths["router"],
            "The target-free portfolio router is not deployable.",
            "Case-holdout accuracy remains below the fail-closed threshold.",
        ),
    ]
    return {
        "version": "chip-postfix-claim-registry-v1",
        "status": "pass_cpu_registry" if not missing else "fail_missing_evidence",
        "scope": "CPU evidence registry; no new inference or route run.",
        "sources": source_paths,
        "missing_sources": missing,
        "claims": claims,
        "invalidated": {
            "pre_fix_m48_three_seed": {
                "evidence": source_paths["invalidated_portfolio"],
                "reason": "Fixed-shape blocks were incorrectly position-locked.",
            },
            "accidental_gpu_smoke": {
                "evidence": [
                    "outputs/reports/chip_v2_gpu_readiness_v1.json",
                    "outputs/reports/chip_v2_mechanism_smoke_v1.json",
                ],
                "reason": "The user constrained this goal to no GPU experiments.",
            },
        },
        "metrics": {
            "paired_cost_relative": paired["cost"]["relative_delta"],
            "paired_v_rel_relative": paired["v_rel"]["relative_delta"],
            "grouping_relative": paired["grouping"]["relative_delta"],
            "boundary_relative": paired["boundary"]["relative_delta"],
            "b2b_relative": paired["b2b_hpwl_gap"]["relative_delta"],
            "area_relative": paired["area_gap"]["relative_delta"],
            "rho_relative": paired["rho_mae"]["relative_delta"],
        },
        "sota_ready": False,
        "matched_budget_anchor_found": False,
        "gpu_experiments_run": False,
    }


def render_document(registry: dict) -> str:
    lines = [
        "# Chip Evidence Lineage: Post-fix",
        "",
        "CPU registry only. No GPU quality or ORFS result is promoted.",
        "",
        "## Claims",
        "",
        "| Status | Claim | Boundary |",
        "|---|---|---|",
    ]
    for claim in registry["claims"]:
        lines.append(
            f"| {claim['status']} | {claim['statement']} | "
            f"{claim['boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Invalidated",
            "",
            "- The pre-fix M48 three-seed portfolio is invalid for quality comparison.",
            "- The accidental GPU readiness/smoke is excluded from all evidence.",
            "",
            "## Decision",
            "",
            (
                "The registry does not support SOTA, same-inference superiority, "
                "or external routed-PPA transfer."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    registry = build_registry(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.document.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    args.document.write_text(render_document(registry), encoding="utf-8")
    print(args.output)
    if registry["status"] != "pass_cpu_registry":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

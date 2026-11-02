#!/usr/bin/env python3
"""Classify historical chip mechanisms as closed, reusable, or reopened."""

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
        default=ROOT / "outputs/reports/chip_architecture_salvage_matrix_v1.json",
    )
    parser.add_argument(
        "--document",
        type=Path,
        default=ROOT / "docs/chip_architecture_salvage_review_v2.md",
    )
    return parser.parse_args()


def build_matrix(root: Path) -> dict:
    entries = [
        {
            "family": "M36-M39 hidden hierarchy bridge and bbox consensus",
            "decision": "closed",
            "evidence": [
                "outputs/reports/chip_m36_m39_architecture_screen_v1.json",
                "docs/chip_m36_m39_architecture_screen_v1.md",
            ],
            "failure": (
                "Grouping or V_rel gains did not survive joint legalized cost; "
                "b2b and p2b regressed sharply."
            ),
            "reusable": [
                "membership-aware hierarchy pooling",
                "zero-initialized residual gating pattern",
            ],
            "reuse_target": "M50 same-call state commitment, not the rejected residual.",
            "rerun_allowed": False,
        },
        {
            "family": "M40 self-conditioned flow",
            "decision": "closed",
            "evidence": [
                "outputs/reports/chip_m40_self_conditioned_quick_refiner400_dev32.json",
                "docs/chip_m40_self_conditioned_flow_v1.md",
            ],
            "failure": (
                "Training and raw metrics improved, but high-complexity cases "
                "escalated legalizer rung and solve counts."
            ),
            "reusable": ["state history plumbing", "legalizer rung/solve diagnostics"],
            "reuse_target": "Runtime diagnostics only; do not reopen keep-probability sweeps.",
            "rerun_allowed": False,
        },
        {
            "family": "M45 projector-displacement output head",
            "decision": "closed",
            "evidence": [
                "outputs/reports/chip_m45_projector_residual_quick_refiner400_dev32.json",
                "docs/chip_m45_m46_model_structure_repairs_v1.md",
            ],
            "failure": "The auxiliary head regressed cost, V_rel, b2b and p2b.",
            "reusable": ["hard-projector displacement targets"],
            "reuse_target": "M53 endpoint compatibility, without projected-output substitution.",
            "rerun_allowed": False,
        },
        {
            "family": "M46 all-layer state-conditioned edge bias",
            "decision": "closed",
            "evidence": [
                "outputs/reports/chip_m46b_state_edge_normalized_quick_refiner400_dev32.json",
                "docs/chip_m45_m46_model_structure_repairs_v1.md",
            ],
            "failure": "Proxy gains did not survive legalization and joint primary metrics.",
            "reusable": ["active b2b/p2b geometry features", "row-normalized edge aggregation"],
            "reuse_target": "M52 current-state dynamic graph input with relation-specific fusion.",
            "rerun_allowed": False,
        },
        {
            "family": "M10 final-only projection-consistent flow",
            "decision": "closed",
            "evidence": [
                "outputs/reports/chip_m10_projection_consistent_flow_final_only_confirmation_fallback_cpu_v1.json"
            ],
            "failure": "Quick gains reversed at refiner1600.",
            "reusable": ["final-only projection diagnostic"],
            "reuse_target": "M53 endpoint validation only.",
            "rerun_allowed": False,
        },
        {
            "family": "U1 additive dynamic graph relay",
            "decision": "closed",
            "evidence": ["outputs/reports/chip_u1_dynamic_graph_screen_v1.json"],
            "failure": "Confirmation regressed V_rel.",
            "reusable": ["StateEdgeRelay edge geometry encoding"],
            "reuse_target": "M52 dynamic graph-input conditioning.",
            "rerun_allowed": False,
        },
        {
            "family": "Spatial occupancy, port consensus and ordered group staging",
            "decision": "reusable_component",
            "evidence": [
                "outputs/reports/chip_architecture_v2_contract_audit.json",
                "chipstodo.json",
            ],
            "failure": (
                "Standalone residual or ordering hypotheses lacked a joint legalized gain."
            ),
            "reusable": [
                "differentiable rasterization",
                "sparse incidence aggregation",
                "group-priority diagnostics",
            ],
            "reuse_target": "M52/M55 typed graph and routeability fields only.",
            "rerun_allowed": False,
        },
    ]
    missing = sorted(
        {
            path
            for entry in entries
            for path in entry["evidence"]
            if not (root / path).is_file()
        }
    )
    return {
        "version": "chip-architecture-salvage-matrix-v1",
        "status": "pass_cpu_audit" if not missing else "fail_missing_evidence",
        "scope": "Historical chip architecture failure triage; no new experiment.",
        "entries": entries,
        "missing_evidence": missing,
        "policy": (
            "Closed output contracts are not swept again. Reusable components may "
            "enter M50-M55 only through a new causal hypothesis and matched ablation."
        ),
        "gpu_experiments_run": False,
    }


def render_document(matrix: dict) -> str:
    lines = [
        "# Chip Architecture Salvage Review v2",
        "",
        "Status: CPU evidence audit only. No GPU or ORFS experiment was run.",
        "",
        "The matrix separates failed output contracts from reusable operators.",
        "A reusable component is not evidence that its old mechanism worked.",
        "",
        "| Family | Decision | Reuse target |",
        "|---|---|---|",
    ]
    for entry in matrix["entries"]:
        lines.append(
            f"| {entry['family']} | {entry['decision']} | "
            f"{entry['reuse_target']} |"
        )
    lines.extend(
        [
            "",
            "## Rules",
            "",
            "- Do not rerun a closed gate with width, gate, mask, or scale sweeps.",
            "- Preserve legalizer rung/solve counts in every future mechanism probe.",
            "- Reopen only with a new state contract and a matched control.",
            "",
            f"Machine-readable matrix: `{matrix['status']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    matrix = build_matrix(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.document.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    args.document.write_text(render_document(matrix), encoding="utf-8")
    print(args.output)
    if matrix["status"] != "pass_cpu_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

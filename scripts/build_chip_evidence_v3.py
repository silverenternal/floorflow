#!/usr/bin/env python3
"""Promote completed chip diagnostics into the v3 evidence ledger."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_cvpr_evidence_v3.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = copy.deepcopy(
        read("outputs/reports/chip_token_flow_claim_gate_v3.json")
    )
    triage = read("outputs/reports/chip_metric_triage_v1.json")
    tail_budget = read("outputs/reports/chip_tail_budget_mechanism_v1.json")
    gap = read("outputs/reports/chip_trinity_gap_decomposition_v1.json")
    evidence.update(
        {
            "status": "complete",
            "scope": (
                "FloorSet-Lite official-100 matched chip diagnostics; "
                "separate from the floorplan claim"
            ),
            "claim_status": "scoped_mechanism_and_budget_evidence_complete",
            "allowed_main_claim": (
                "Under the same CFM objective and exact-area decoder, "
                "per-block graph-token transport improves structural geometry "
                "and reduces structural-failure cost tails relative to the "
                "matched global-slot control. It reaches the declared pooled "
                "quality threshold with fewer measured NFE or refiner steps."
            ),
            "allowed_boundary": (
                "The candidate remains behind the same-protocol Trinity d192 "
                "comparator on aggregate cost and V_rel. No FloorSet or "
                "complete-CAD SOTA claim is supported."
            ),
            "metric_triage": triage,
            "tail_budget_mechanism": tail_budget,
            "trinity_gap_decomposition": gap,
            "promoted_evidence": [
                "Matched structural gains on b2b connectivity, area/outline, grouping, and rho/aspect.",
                "Cost improvement concentrated in structural-failure tails.",
                "Equal-quality NFE/refiner budget diagnostic, separate from the primary endpoint.",
                "Exact Vrel and log-cost decomposition with disclosed source residual.",
            ],
            "demoted_evidence": triage["demote_no_claim"],
            "required_boundaries": list(
                dict.fromkeys(
                    evidence.get("required_boundaries", [])
                    + [
                        "Do not claim uniform improvement.",
                        "Do not claim monotonic advantage through the extreme V_rel tail.",
                        "Do not replace NFE4 + 400 with the equal-quality budget diagnostic.",
                        "Do not claim stable high-b2b or high-pin stratum gains.",
                        "Do not turn component decomposition into a single-factor causal claim.",
                    ]
                )
            ),
            "package_target": "outputs/cvpr_package_v13_chips",
        }
    )
    checks = {
        "claim_gate_selected": all(evidence["checks"].values()),
        "metric_triage_complete": triage["status"] == "complete",
        "metric_triage_checks": all(triage["checks"].values()),
        "tail_budget_pass": tail_budget["status"] == "pass",
        "tail_budget_checks": all(tail_budget["checks"].values()),
        "trinity_decomposition_pass": gap["status"] == "pass",
        "trinity_decomposition_checks": all(gap["checks"].values()),
        "primary_endpoint_unchanged": (
            tail_budget["budget_evidence"]["primary_endpoint"]
            == "NFE4 + 400 refiner steps"
        ),
        "no_chip_sota_claim": gap["interpretation"]["sota"] == "not claimed",
    }
    evidence["synthesis_checks"] = checks
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"chip v3 gates failed: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

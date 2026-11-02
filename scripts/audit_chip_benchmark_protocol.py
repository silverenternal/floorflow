#!/usr/bin/env python3
"""Audit chip benchmark comparability before external-baseline claims."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.scoring.total import (
    total_blockcount,
    total_exp,
    total_mean,
)

INTERNAL_CASE_FILES = {
    "flat_dense_seed19": "outputs/chip_large/v2_20k/dense_no_ot_huber/official_test100_nfe8_legalized_cases.json",
    "flat_dense_seed20": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s20/official_test100_nfe8_legalized_cases.json",
    "flat_dense_seed21": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s21/official_test100_nfe8_legalized_cases.json",
    "global_slot_seed19": "outputs/chip_slot/global_v1/global_slot_overlap10_20k/official_test100_nfe8_legalized_cases.json",
    "global_slot_seed20": "outputs/chip_slot/global_v2/global_slot_seed20_overlap10_20k/official_test100_nfe8_legalized_cases.json",
    "global_slot_seed21": "outputs/chip_slot/global_v2/global_slot_seed21_overlap10_20k/official_test100_nfe8_legalized_cases.json",
    "fixed_hierarchy_seed19": "outputs/chip_hierarchical/v2_matched/hierarchical_slot_overlap10_20k/official_test100_nfe8_legalized_cases.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "outputs/reports/chip_cvpr_evidence_ledger_v1.json",
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=ROOT / "outputs/reports/trinity_d192_official_test100.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_benchmark_protocol_audit_v1.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_summary(
    costs: list[float],
    block_counts: list[int],
    v_rel: list[float],
) -> dict:
    if not costs:
        return {"cases": 0}
    return {
        "cases": len(costs),
        "cost_mean": total_mean(costs, block_counts),
        "cost_exp_weighted": total_exp(costs, block_counts),
        "cost_blockcount_weighted": total_blockcount(costs, block_counts),
        "cost_min": min(costs),
        "cost_max": max(costs),
        "v_rel_mean": sum(v_rel) / len(v_rel),
    }


def internal_summary(path: Path) -> dict:
    rows = read_json(path)
    return scalar_summary(
        [float(row["legalized_hard_cost"]) for row in rows],
        [int(row["blocks"]) for row in rows],
        [float(row["legalized_v_rel"]) for row in rows],
    )


def external_summary(path: Path) -> dict:
    report = read_json(path)
    rows = report["rows"]
    summary = scalar_summary(
        [float(row["cost"]) for row in rows],
        [int(row["blocks"]) for row in rows],
        [float(row["v_rel"]) for row in rows],
    )
    summary["feasible_fraction"] = float(
        sum(bool(row["feasible"]) for row in rows) / len(rows)
    )
    return summary


def main() -> None:
    args = parse_args()
    ledger = read_json(args.ledger)
    internal = {}
    for name, relative in INTERNAL_CASE_FILES.items():
        path = ROOT / relative
        internal[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "summary": internal_summary(path) if path.is_file() else None,
        }
    external = external_summary(args.external)
    protocols = {
        "internal_flow": {
            "source": "current evaluate_chip_official.py reports",
            "sampler_steps": 8,
            "refiner": None,
            "legalizer": "scale_pack",
            "scorer": "score_full_fast",
            "samples_per_case": 1,
            "best_of_n_used": False,
            "ema": False,
        },
        "external_trinity_d192": {
            "source": "existing eval_trinity_baseline.py report",
            "sampler_steps": 16,
            "refiner": "closed",
            "refiner_steps": 400,
            "legalizer": "scale_pack default route",
            "scorer": "validate(full)",
            "samples_per_case": 1,
            "best_of_n_used": False,
            "ema": "release checkpoint loader",
            "provenance_serialized": False,
        },
    }
    mismatches = [
        "sampling_steps",
        "refiner",
        "scorer",
        "ema_handling",
        "provenance_serialized",
    ]
    diagnostic_gap = {
        "cost_mean_absolute": (
            external["cost_mean"] - internal["global_slot_seed19"]["summary"]["cost_mean"]
        ),
        "cost_mean_relative_to_external": (
            external["cost_mean"] - internal["global_slot_seed19"]["summary"]["cost_mean"]
        )
        / external["cost_mean"],
        "v_rel_absolute": (
            external["v_rel_mean"]
            - internal["global_slot_seed19"]["summary"]["v_rel_mean"]
        ),
    }
    checks = {
        "ledger_pass": ledger.get("status") == "pass",
        "internal_rows_complete": all(
            item["exists"] and item["summary"]["cases"] == 100
            for item in internal.values()
        ),
        "external_rows_complete": external["cases"] == 100,
        "direct_comparison_prohibited": bool(mismatches),
        "external_protocol_provenance_incomplete": True,
    }
    report = {
        "status": (
            "pass_with_protocol_separation"
            if all(checks.values())
            else "fail"
        ),
        "scope": "protocol audit; diagnostic gaps are not paper-table claims",
        "checks": checks,
        "protocols": protocols,
        "mismatches": mismatches,
        "internal": internal,
        "external": {
            "path": str(args.external),
            "summary": external,
        },
        "diagnostic_gap_not_for_claim": diagnostic_gap,
        "decision": (
            "Do not compare the current internal rows directly against the "
            "existing Trinity report or claim SOTA. P3 must run a matched "
            "refiner/NFE/scorer experiment."
        ),
        "verified_online_sources": [
            {
                "title": "FloorSet paper",
                "url": "https://arxiv.org/abs/2405.05480",
                "verified_on": "2026-10-26",
            },
            {
                "title": "Trinity paper",
                "url": "https://arxiv.org/abs/2610.04957",
                "verified_on": "2026-10-26",
            },
            {
                "title": "Trinity official code",
                "url": "https://github.com/Kohaku-Lab/Trinity",
                "verified_on": "2026-10-26",
            },
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "mismatches": mismatches,
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

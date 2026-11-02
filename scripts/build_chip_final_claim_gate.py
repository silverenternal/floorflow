#!/usr/bin/env python3
"""Freeze the supported route-B claim after all chip experiments."""

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
        default=ROOT / "outputs/reports/chip_final_claim_gate_v1.json",
    )
    return parser.parse_args()


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    final = read("outputs/reports/chip_final_three_seed_evidence_v1.json")
    mechanism = read("outputs/reports/chip_graph_mechanism_ablation_v1.json")
    architecture = read(
        "outputs/reports/chip_output_parameterization_and_dropout_v1.json"
    )
    range_result = read("outputs/reports/chip_range_warm_restart_v1.json")
    external = read("outputs/reports/trinity_d192_matched_protocol_v2.json")
    checks = {
        "three_seed_replication_passed": final["status"] == "pass",
        "mechanism_attribution_passed": mechanism["status"] == "pass",
        "simpler_velocity_model_retained": (
            architecture["status"] == "keep_velocity_full_graph"
        ),
        "range_adaptation_kept_negative": (
            range_result["status"] == "reject_range_adaptation"
        ),
        "external_comparator_reported": True,
    }
    selected = all(checks.values())
    claim = {
        "id": "B_graph_global_slot" if selected else "B_global_slot_fallback",
        "statement": (
            "A graph-conditioned invariant global-latent flow with exact-area "
            "decoding improves constrained chip floorplanning over a "
            "parameter-matched global-slot control; both rich conditioning and "
            "graph topology are required."
            if selected
            else "The existing global-slot result remains the defensible fallback."
        ),
        "supported_scope": [
            "FloorSet-Lite official-100",
            "fixed data, optimizer schedule, and frozen training runtime",
            "raw NFE8 and NFE4 plus the shared 400-step refiner",
            "three seeds with paired official-case statistics",
        ],
        "negative_controls": [
            "parameter-matched global slot",
            "features-only and topology-only graph ablations",
            "absolute and residual slot transport",
            "x0, conditioning dropout, and x0+dropout",
            "fixed hierarchy",
            "raw-only range adaptation",
        ],
        "limitations": [
            "The current graph model remains worse than the same-protocol Trinity d192 checkpoint.",
            "Raw feasibility is zero before scale-pack legalization.",
            "The range-adapted model improves raw long cases but harms refined endpoints.",
            "MIB groups are exercised by physics parity but are rare in the official-100 endpoint.",
        ],
        "external_same_protocol": {
            "trinity_hard_cost": external["mean_cost"],
            "trinity_v_rel": external["mean_v_rel"],
            "final_graph_mean_refined_hard_cost": final[
                "seed_mean_absolute"
            ]["refined"]["graph"]["legalized_hard_cost"],
            "final_graph_mean_refined_v_rel": final["seed_mean_absolute"][
                "refined"
            ]["graph"]["legalized_v_rel"],
            "final_control_mean_refined_hard_cost": final[
                "seed_mean_absolute"
            ]["refined"]["control"]["legalized_hard_cost"],
            "final_control_mean_refined_v_rel": final["seed_mean_absolute"][
                "refined"
            ]["control"]["legalized_v_rel"],
            "graph_hard_cost_gap_to_trinity": (
                final["seed_mean_absolute"]["refined"]["graph"][
                    "legalized_hard_cost"
                ]
                / external["mean_cost"]
                - 1.0
            ),
        },
    }
    report = {
        "status": "selected" if selected else "fallback",
        "checks": checks,
        "claim": claim,
        "evidence": {
            "three_seed": "outputs/reports/chip_final_three_seed_evidence_v1.json",
            "mechanism": "outputs/reports/chip_graph_mechanism_ablation_v1.json",
            "architecture_choices": (
                "outputs/reports/chip_output_parameterization_and_dropout_v1.json"
            ),
            "range": "outputs/reports/chip_range_warm_restart_v1.json",
            "screen": "outputs/reports/chip_graph_slot_20k_v1.json",
            "external": "outputs/reports/trinity_d192_matched_protocol_v2.json",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

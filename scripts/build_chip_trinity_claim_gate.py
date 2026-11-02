#!/usr/bin/env python3
"""Freeze the supported claim for the three-seed graph-token flow result."""

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
        default=ROOT / "outputs/reports/chip_trinity_claim_gate_v1.json",
    )
    return parser.parse_args()


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    final = read("outputs/reports/chip_trinity_token_three_seed_v1.json")
    parity = read("outputs/reports/chip_trinity_parity_gap_v1.json")
    external = read("outputs/reports/trinity_d192_matched_protocol_v2.json")
    mechanism = read("outputs/reports/chip_graph_mechanism_ablation_v1.json")
    local_slots = read("outputs/reports/chip_condition_slot_identity_probe_v2.json")
    checks = {
        "three_seed_both_metrics_all_seeds": final["checks"][
            "candidate_wins_both_refined_metrics_all_seeds"
        ],
        "seed_mean_refined_both": final["checks"][
            "seed_mean_refined_improves_both"
        ],
        "parameter_match_within_1pct": (
            parity["gap"]["parameter_relative"] < 0.01
        ),
        "graph_mechanism_retained": mechanism["status"] == "pass",
        "local_slot_route_retained_negative": (
            local_slots["status"] == "reject_or_redesign"
        ),
        "external_same_protocol_available": (
            float(external["mean_cost"]) > 0.0
            and float(external["mean_v_rel"]) > 0.0
        ),
    }
    selected = all(checks.values())
    candidate = final["summary"]["refined"]["absolute"]["candidate"]
    baseline = final["summary"]["refined"]["absolute"]["baseline"]
    report = {
        "status": "selected" if selected else "blocked",
        "checks": checks,
        "claim": {
            "id": "B_graph_token_flow",
            "statement": (
                "A same-parameter graph-token flow, built by adapting the pinned "
                "Trinity denoiser to the existing chip objective and exact-area "
                "decoder, improves both official endpoints over the implemented "
                "graph-conditioned invariant-global-slot decoder across three "
                "seeds after the common refiner."
            ),
            "mechanism": (
                "The result follows the verified architecture gap: per-block flow "
                "state, spectral graph positional encoding, qk-normalized SDPA, "
                "SwiGLU/RMSNorm blocks, graph message mixing, and token time "
                "conditioning."
            ),
            "supported_scope": [
                "FloorSet-Lite official-100",
                "identical 50k cache, optimizer schedule, batch size, and seed set",
                "raw NFE8 and NFE4 plus the shared 400-step refiner",
                "seed-mean and per-seed legalized hard cost and V_rel",
                "parameter match within 1% to the established graph decoder",
            ],
            "primary_effect": {
                "refined_seed_mean_relative": final["summary"]["refined"][
                    "seed_mean_relative_delta"
                ],
                "raw_seed_mean_relative": final["summary"]["raw"][
                    "seed_mean_relative_delta"
                ],
                "candidate_refined": candidate,
                "baseline_refined": baseline,
            },
            "external_same_protocol": {
                "trinity_hard_cost": external["mean_cost"],
                "trinity_v_rel": external["mean_v_rel"],
                "candidate_hard_cost": candidate["legalized_hard_cost"],
                "candidate_v_rel": candidate["legalized_v_rel"],
                "candidate_hard_cost_gap_to_trinity": (
                    candidate["legalized_hard_cost"]
                    / float(external["mean_cost"])
                    - 1.0
                ),
                "candidate_v_rel_gap_to_trinity": (
                    candidate["legalized_v_rel"]
                    / float(external["mean_v_rel"])
                    - 1.0
                ),
            },
            "not_supported": [
                "FloorSet state of the art",
                "Superiority over the same-protocol Trinity d192 checkpoint",
                "A useful invariant global flow state; the prior causal audit rejected it",
                "Learned stable local-slot identity or predictable residual transport",
                "Stable paired gains on every individual official case",
                "Raw placement feasibility without scale-pack legalization",
            ],
            "statistical_caveats": [
                "Paired case-level intervals include zero on seeds 19 and 21 for refined metrics.",
                "Seed 20 has paired intervals below zero for both refined endpoints.",
                "Report the seed-mean effect together with per-seed and long-tail intervals.",
            ],
        },
        "evidence": {
            "primary": "outputs/reports/chip_trinity_token_three_seed_v1.json",
            "parity": "outputs/reports/chip_trinity_parity_gap_v1.json",
            "external": "outputs/reports/trinity_d192_matched_protocol_v2.json",
            "graph_mechanism": "outputs/reports/chip_graph_mechanism_ablation_v1.json",
            "local_slot_negative": (
                "outputs/reports/chip_condition_slot_identity_probe_v2.json"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

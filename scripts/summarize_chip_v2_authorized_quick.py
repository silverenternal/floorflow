#!/usr/bin/env python3
"""Summarize authorized chip v2 quick screens and failure mechanisms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_hpwl_gap",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_rho_mae",
)

RAW_METRICS = (
    "total",
    "latent_mse",
    "rho_mae",
    "overlap",
    "group",
    "boundary",
    "wl",
    "area",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v2_authorized_quick_summary_v1.json",
    )
    return parser.parse_args()


def load(path: str | Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def relative(values: dict, baseline: dict, keys: tuple[str, ...]) -> dict:
    result = {}
    for key in keys:
        base = float(baseline[key])
        value = float(values[key])
        result[key] = {
            "control": base,
            "candidate": value,
            "relative": (value / base - 1.0) if base else None,
        }
    return result


def repair_row(
    stage: str,
    decision: str,
    failure: str,
    repair: str | list[str],
    shared_control_train: dict,
    shared_control_eval: dict,
) -> dict:
    train_path = (
        f"outputs/chip_hierarchy/v2_authorized_quick/{stage.lower()}/candidate/"
        f"{stage.lower()}_candidate_quick.json"
    )
    legalized_path = (
        f"outputs/reports/chip_{stage.lower()}_authorized_quick_refiner400_dev32_v1.json"
    )
    train = load(train_path)
    legalized = load(legalized_path)["results"]["nfe4_refine400"]
    return {
        "decision": decision,
        "failure": failure,
        "repair": [repair] if isinstance(repair, str) else repair,
        "checkpoint": train["checkpoint_selection"]["selected_checkpoint"],
        "parameters": train["model_parameters"],
        "train_budget": train["execution_budget"],
        "training_wall_seconds": float(train["history"][-1]["elapsed_seconds"]),
        "raw": relative(train["final_dev"], shared_control_train["final_dev"], RAW_METRICS),
        "legalized": relative(legalized, shared_control_eval, METRICS),
        "legalized_absolute": legalized,
        "report_paths": {"train": train_path, "legalized": legalized_path},
    }


def build_summary() -> dict:
    shared_control_train = load(
        "outputs/chip_hierarchy/v2_authorized_quick/m50/control/m50_control_quick.json"
    )
    shared_control_eval = load(
        "outputs/reports/chip_v2_shared_control_refiner400_dev32_v1.json"
    )["results"]["nfe4_refine400"]
    stages = {
        "M50": {
            "train": "outputs/chip_hierarchy/v2_authorized_quick/m50/candidate/m50_candidate_quick.json",
            "legalized": "outputs/reports/chip_m50_authorized_quick_refiner400_dev32_v1.json",
            "decision": "reject",
            "failure": (
                "Coarse commitment followed by relation supervision increased raw "
                "total and did not survive legalization; p2b remained severely worse."
            ),
            "repair": [
                "Replace hard coarse freezing with a confidence-gated bounded anchor.",
                "Commit only low-risk clusters whose membership is stable across trajectory samples.",
                "Add an explicit p2b displacement penalty to the relation head.",
            ],
        },
        "M51": {
            "train": "outputs/chip_hierarchy/v2_authorized_quick/m51/candidate/m51_candidate_quick.json",
            "legalized": "outputs/reports/chip_m51_authorized_quick_refiner400_dev32_v1.json",
            "decision": "reject",
            "failure": (
                "Center/rho factorization improved raw rho slightly but raised "
                "legalized V_rel and grouping while p2b exploded."
            ),
            "repair": [
                "Share a latent trunk between center and rho heads and add an exact-area consistency loss.",
                "Gate shape updates on connected-pin pressure and block boundary state.",
                "Add a center-rho cross-consistency target from the exact decoder.",
            ],
        },
        "M53": {
            "train": "outputs/chip_hierarchy/v2_authorized_quick/m53/candidate/m53_candidate_quick.json",
            "legalized": "outputs/reports/chip_m53_authorized_quick_refiner400_dev32_v1.json",
            "decision": "reject_legalizer_inversion",
            "failure": (
                "Best raw total and lower V_rel, but legalized cost, area, b2b and "
                "especially p2b regressed. The endpoint compatibility objective is "
                "optimizing a refiner energy that does not predict legalizer behavior."
            ),
            "repair": [
                "Replace generic refiner energy with rung/solve and per-block displacement constraints.",
                "Add legalized-cost and area-gap proxies to the endpoint objective.",
                "Reject checkpoints by legalized validation, never by raw curve total.",
            ],
        },
        "M52": {
            "train": "outputs/chip_hierarchy/v2_authorized_quick/m52/candidate/m52_candidate_quick.json",
            "legalized": "outputs/reports/chip_m52_authorized_quick_refiner400_dev32_v1.json",
            "decision": "reject",
            "failure": (
                "Relation-specific graph conditioning increased legalized cost, "
                "V_rel, grouping and p2b. Port incidence is likely dominating the "
                "relation path despite typed masks."
            ),
            "repair": [
                "Normalize each relation family by active incidence and add per-family residual trust radii.",
                "Add a relation-gate entropy floor and a port-incidence ablation during training.",
                "Route port evidence only through block queries, never global or structural queries.",
            ],
        },
        "M55": {
            "train": "outputs/chip_hierarchy/v2_authorized_quick/m55/candidate/m55_candidate_quick.json",
            "legalized": "outputs/reports/chip_m55_authorized_quick_refiner400_dev32_v1.json",
            "decision": "reject_primary_but_split",
            "failure": (
                "Route plus source-schedule reduced cost, b2b and p2b but worsened "
                "V_rel, grouping and rho. Two mechanisms are confounded in one arm."
            ),
            "repair": [
                "Split route-field-only and source-schedule-only matched arms.",
                "Keep the route field only if it preserves V_rel and grouping after legalization.",
                "Add a target-free routing proxy loss or test-time displacement before fusion.",
            ],
        },
    }
    rows = {}
    for name, stage in stages.items():
        train = load(stage["train"])
        legalized = load(stage["legalized"])["results"]["nfe4_refine400"]
        rows[name] = {
            "decision": stage["decision"],
            "failure": stage["failure"],
            "repair": stage["repair"],
            "checkpoint": train["checkpoint_selection"]["selected_checkpoint"],
            "parameters": train["model_parameters"],
            "train_budget": train["execution_budget"],
            "training_wall_seconds": float(train["history"][-1]["elapsed_seconds"]),
            "raw": relative(
                train["final_dev"],
                shared_control_train["final_dev"],
                RAW_METRICS,
            ),
            "legalized": relative(legalized, shared_control_eval, METRICS),
            "legalized_absolute": legalized,
        }
    m54_control = load(
        "outputs/reports/chip_m54_control_authorized_quick_refiner400_dev32_v1.json"
    )["results"]["nfe4_refine400"]
    m54_candidate = load(
        "outputs/reports/chip_m54_authorized_quick_refiner400_dev32_v1.json"
    )["results"]["nfe4_refine400"]
    rows["M54"] = {
        "decision": "reject_subset_overfit",
        "failure": (
            "The 32-case response sidecar caused severe overfit. M54 improved over "
            "its matched 32-case control but both were far worse than the shared "
            "5k control, and p2b remained pathological."
        ),
        "repair": [
            "Build a stratified 1k-case legalizer-response sidecar before any retrain.",
            "Decay the response head after warmup and regularize rung predictions.",
            "Require response-head validation on cases excluded from sidecar construction.",
        ],
        "checkpoint": load(
            "outputs/chip_hierarchy/v2_authorized_quick/m54/candidate/m54_candidate_quick.json"
        )["checkpoint_selection"]["selected_checkpoint"],
        "parameters": load(
            "outputs/chip_hierarchy/v2_authorized_quick/m54/candidate/m54_candidate_quick.json"
        )["model_parameters"],
        "raw_vs_shared_control": relative(
            load("outputs/chip_hierarchy/v2_authorized_quick/m54/candidate/m54_candidate_quick.json")[
                "final_dev"
            ],
            shared_control_train["final_dev"],
            RAW_METRICS,
        ),
        "legalized_vs_shared_control": relative(
            m54_candidate,
            shared_control_eval,
            METRICS,
        ),
        "legalized_vs_matched_32case_control": relative(
            m54_candidate,
            m54_control,
            METRICS,
        ),
        "legalized_absolute": m54_candidate,
        "train_budget": load(
            "outputs/chip_hierarchy/v2_authorized_quick/m54/candidate/m54_candidate_quick.json"
        )["execution_budget"],
        "training_wall_seconds": float(
            load(
                "outputs/chip_hierarchy/v2_authorized_quick/m54/candidate/m54_candidate_quick.json"
            )["history"][-1]["elapsed_seconds"]
        ),
    }
    repair_stages = {
        "M57": {
            "decision": "reject_quick_gate_cost_vrel_p2b",
            "failure": (
                "Confidence gating reduced the damage from hard commitment but cost and "
                "V_rel still rose, while p2b increased more than sixfold."
            ),
            "repair": [
                "Do not tune the coarse-anchor family further before source adaptation.",
                "Keep the bounded-anchor contract for a later ablation only after a primary gate passes.",
            ],
        },
        "M58": {
            "decision": "reject_quick_gate_cost_vrel_p2b",
            "failure": (
                "The shared geometry trunk preserved cost better than M57 but V_rel, area "
                "and b2b regressed, and p2b again became pathological."
            ),
            "repair": [
                "Treat factorized geometry as a failed independent mechanism.",
                "Revisit only with a legalizer-conditioned shape target after source adaptation is fixed.",
            ],
        },
        "M59": {
            "decision": "reject_quick_gate_legalized_cost_area_b2b_p2b",
            "failure": (
                "Endpoint constraints improved V_rel by 1.92% but cost rose 5.08%, with "
                "large area, b2b and p2b reversals after legalization."
            ),
            "repair": [
                "Do not open M62: its required endpoint parent did not pass.",
                "Move legalizer feedback to confirmation-time diagnostics instead of an internal training loss.",
            ],
        },
        "M60": {
            "decision": "reject_quick_gate_cost_area_b2b_p2b",
            "failure": (
                "Incidence normalization improved V_rel by 2.11% but cost rose 3.45%, "
                "area and b2b worsened by 29% and 51%, and p2b worsened 7.37x."
            ),
            "repair": [
                "Port evidence remains unsafe even after incidence normalization.",
                "Keep relation-specific graph code as an ablation, not in the promoted path.",
            ],
        },
        "M61A": {
            "decision": "reject_quick_gate_cost_vrel_structure",
            "failure": (
                "The route field plus proxy raised cost 3.44% and V_rel 27.02%; the route "
                "mechanism damaged grouping, boundary and p2b."
            ),
            "repair": [
                "Retire target-free route-proxy training for the current exact-area flow.",
                "Retain only external routed metrics as an integration test after internal selection.",
            ],
        },
        "M61B": {
            "decision": "reject_quick_gate_vrel_grouping",
            "failure": (
                "The confounded source-only arm reduced cost, area, b2b and p2b but raised "
                "V_rel 2.99% and grouping 23.08%."
            ),
            "repair": [
                "Remove backbone unfreezing and dihedral augmentation to isolate the clean source schedule.",
            ],
        },
        "M64": {
            "decision": "promote_to_confirmation",
            "failure": (
                "No quick-gate failure: the clean mixed/derived source schedule jointly "
                "reduced cost and V_rel with feasibility 1.0."
            ),
            "repair": [
                "Run 5k matched control/candidate confirmation; 2k training did not reach the final source probability.",
                "Evaluate NFE4/refiner400 and the primary NFE4/refiner1600 endpoint before any SOTA claim.",
            ],
        },
        "M65": {
            "decision": "reject_quick_gate_vrel",
            "failure": (
                "Adding incidence-normalized graph conditioning to the clean source schedule "
                "erased most of the cost gain and raised V_rel instead of improving M64."
            ),
            "repair": [
                "Select M64 as the confirmation candidate and leave graph-source fusion out.",
                "Do not infer that graph conditioning is globally useless; it failed under this training budget and implementation.",
            ],
        },
    }
    for name, stage in repair_stages.items():
        rows[name] = repair_row(
            name,
            stage["decision"],
            stage["failure"],
            stage["repair"],
            shared_control_train,
            shared_control_eval,
        )
    return {
        "version": "chip-v2-authorized-quick-summary-v1",
        "status": "joint_primary_candidate_selected_confirmation_pending",
        "device": "cuda:0",
        "training_subset_cases": 5000,
        "validation_cases": 32,
        "nfe": 4,
        "refiner_steps": 400,
        "training_wall_seconds": {
            name: row["training_wall_seconds"] for name, row in rows.items()
        },
        "shared_control": shared_control_eval,
        "shared_control_raw": shared_control_train["final_dev"],
        "stages": rows,
        "selected_candidate": {
            "stage": "M64",
            "mechanism": "clean mixed/derived source schedule",
            "control_cost_relative": rows["M64"]["legalized"]["legalized_hard_cost"]["relative"],
            "control_v_rel_relative": rows["M64"]["legalized"]["legalized_v_rel"]["relative"],
            "feasible_fraction": rows["M64"]["legalized_absolute"]["legalized_feasible_fraction"],
            "confirmation_required": True,
            "reason": (
                "Only M64 passed the joint cost/V_rel and feasibility quick gate. "
                "M65 is rejected; M62 remains blocked because its required endpoint parent failed."
            ),
        },
        "next_sequence": [
            "Train M64 and matched control for 5,000 steps on the frozen 5k subset.",
            "Evaluate confirmation at NFE4/refiner400 before spending on refiner1600.",
            "Run the NFE4/refiner1600 primary endpoint on the frozen 32-case screen.",
            "Compare M64 with equal-budget Trinity d192 only after the internal gate passes.",
            "Certify multiple seeds and external routed-PPA behavior before making a SOTA claim.",
        ],
        "sota_ready": False,
    }


def main() -> None:
    args = parse_args()
    report = build_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

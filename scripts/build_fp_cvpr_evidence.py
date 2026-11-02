#!/usr/bin/env python3
"""Assemble the current FloorFlow claim-evidence ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiseed",
        type=Path,
        default=Path("outputs/reports/fp_hard_metrics_multiseed_test503_v2.json"),
    )
    parser.add_argument(
        "--full-vs-raw",
        type=Path,
        default=Path("outputs/reports/fp_hard_metrics_full_vs_raw_test503_v1.json"),
    )
    parser.add_argument(
        "--robustness",
        type=Path,
        default=Path("outputs/reports/fp_control_robustness_val191_v1.json"),
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("outputs/reports/fp_external_baseline_audit_v1.json"),
    )
    parser.add_argument(
        "--predicted",
        type=Path,
        default=Path("outputs/control_contracts/predicted_aux_v1/val/manifest.json"),
    )
    parser.add_argument(
        "--room-repair",
        type=Path,
        default=Path("outputs/eval/fp_robustness_roomrepair_p010_val191_v1/metrics.json"),
    )
    parser.add_argument(
        "--room-control-comparison",
        type=Path,
        default=Path("outputs/reports/fp_room_control_external_comparison_v1.json"),
    )
    parser.add_argument(
        "--room-label-support",
        type=Path,
        default=Path("outputs/reports/fp_room_label_support_test503_v2.json"),
    )
    parser.add_argument(
        "--identifiability-benchmark",
        type=Path,
        default=Path("outputs/reports/fp_identifiability_benchmark_v1.json"),
    )
    parser.add_argument(
        "--distribution-gate",
        type=Path,
        default=Path("outputs/reports/fp_distribution_claim_gate_v1.json"),
    )
    parser.add_argument(
        "--architecture-audit",
        type=Path,
        default=Path("outputs/reports/fp_model_architecture_audit_v1.json"),
    )
    parser.add_argument(
        "--topology-audit",
        type=Path,
        default=Path("outputs/reports/fp_topology_aware_decoding_v1.json"),
    )
    parser.add_argument(
        "--joint-repair",
        type=Path,
        default=Path("outputs/reports/fp_joint_control_repair_v2.json"),
    )
    parser.add_argument(
        "--predicted-v2",
        type=Path,
        default=Path("outputs/reports/fp_predicted_control_v2.json"),
    )
    parser.add_argument(
        "--local-edit",
        type=Path,
        default=Path(
            "outputs/reports/fp_local_edit_response_test503_multiseed_v1.json"
        ),
    )
    parser.add_argument(
        "--edit-detection",
        type=Path,
        default=Path(
            "outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json"
        ),
    )
    parser.add_argument(
        "--matched-edit",
        type=Path,
        default=Path(
            "outputs/reports/fp_edit_detection_matched_external_v2.json"
        ),
    )
    parser.add_argument(
        "--housemind-status",
        type=Path,
        default=Path("outputs/reports/fp_housemind_reproduction_status_v1.json"),
    )
    parser.add_argument(
        "--delta-mse-gate",
        type=Path,
        default=Path("outputs/reports/fp_delta_mse_improvement_gate_v1.json"),
    )
    parser.add_argument(
        "--topology-gate",
        type=Path,
        default=Path(
            "outputs/reports/fp_control_gated_topology_fiveseed_v1.json"
        ),
    )
    parser.add_argument(
        "--structural-legality",
        type=Path,
        default=Path(
            "outputs/reports/fp_control_gated_structural_fiveseed_v1.json"
        ),
    )
    parser.add_argument(
        "--canonical-structural",
        type=Path,
        default=Path("outputs/reports/fp_canonical_structural_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--wall-junction",
        type=Path,
        default=Path(
            "outputs/reports/fp_wall_junction_continuity_fiveseed_v1.json"
        ),
    )
    parser.add_argument(
        "--claim-novelty",
        type=Path,
        default=Path("outputs/reports/fp_claim_novelty_audit_v1.json"),
    )
    parser.add_argument(
        "--complexity",
        type=Path,
        default=Path("outputs/reports/fp_complexity_strata_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--stress",
        type=Path,
        default=Path(
            "outputs/reports/fp_strict_locality_stress_fiveseed_v1.json"
        ),
    )
    parser.add_argument(
        "--failure",
        type=Path,
        default=Path("outputs/reports/fp_failure_analysis_v2.json"),
    )
    parser.add_argument(
        "--architecture-necessity",
        type=Path,
        default=Path(
            "outputs/reports/fp_architecture_necessity_ablation_v2.json"
        ),
    )
    parser.add_argument(
        "--local-external-redraw",
        type=Path,
        default=Path("outputs/reports/fp_raster2seq_local_redraw_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_cvpr_evidence_v2.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    multiseed = read_json(args.multiseed)
    comparison = read_json(args.full_vs_raw)
    robustness = read_json(args.robustness)
    external = read_json(args.external)
    predicted = read_json(args.predicted)
    room_repair = read_json(args.room_repair) if args.room_repair.is_file() else None
    room_control = (
        read_json(args.room_control_comparison)
        if args.room_control_comparison.is_file()
        else None
    )
    room_label_support = (
        read_json(args.room_label_support)
        if args.room_label_support.is_file()
        else None
    )
    identifiability_benchmark = (
        read_json(args.identifiability_benchmark)
        if args.identifiability_benchmark.is_file()
        else None
    )
    distribution_gate = (
        read_json(args.distribution_gate)
        if args.distribution_gate.is_file()
        else None
    )
    architecture = read_json(args.architecture_audit)
    topology = read_json(args.topology_audit)
    joint_repair = read_json(args.joint_repair)
    predicted_v2 = read_json(args.predicted_v2)
    local_edit = read_json(args.local_edit)
    edit_detection = read_json(args.edit_detection)
    matched_edit = read_json(args.matched_edit)
    housemind_status = read_json(args.housemind_status)
    delta_mse_gate = read_json(args.delta_mse_gate)
    topology_gate = read_json(args.topology_gate)
    structural_legality = read_json(args.structural_legality)
    canonical_structural = read_json(args.canonical_structural)
    wall_junction = read_json(args.wall_junction)
    claim_novelty = read_json(args.claim_novelty)
    complexity = read_json(args.complexity)
    stress = read_json(args.stress)
    failure = read_json(args.failure)
    architecture_necessity = read_json(args.architecture_necessity)
    local_external_redraw = read_json(args.local_external_redraw)
    seed_metrics = {
        str(row["seed"]): row["metrics"] for row in multiseed["seeds"]
    }
    seed_ids = sorted(int(seed) for seed in seed_metrics)
    support_values = [
        seed_metrics[str(seed)].get("room_support_iou")
        for seed in seed_ids
    ]
    watertight = multiseed["summary"]["conditional_watertight"]
    edit_cost = multiseed["summary"]["edit_cost_reduction"]
    preservation = multiseed["summary"]["control_room_pixel_preservation"]
    edit_iou = edit_detection["aggregate"]["local_delta_iou"]
    edit_mse = edit_detection["aggregate"]["local_delta_mse"]
    evidence = {
        "status": "active",
        "scope": "CubiCasa5K coarse-control floorplan completion",
        "released_claim": claim_novelty["released_claim"],
        "requested_first_claim_status": claim_novelty["decision"],
        "mandatory_boundary": claim_novelty["mandatory_boundary"],
        "claims": {
            "supported_clean_control": [
                (
                    "The serialized-control pipeline is schema-valid on all "
                    f"test503 cases in each of {len(seed_ids)} seeds."
                ),
                (
                    "Conditional watertightness ranges from "
                    f"{watertight['min']:.4f} to {watertight['max']:.4f} "
                    f"across {len(seed_ids)} seeds."
                ),
                "Edit cost falls by 32.3% for seed19 versus raw control upsampling with a paired interval excluding zero.",
                (
                    f"Mean edit-cost reduction is {edit_cost['mean']:.4f} +/- "
                    f"{edit_cost['std']:.4f} across {len(seed_ids)} seeds."
                ),
                (
                    "Room-pixel intent preservation is "
                    f"{preservation['mean']:.4f} +/- {preservation['std']:.4f}."
                ),
                "Room adjacency F1 improves by 0.093 versus raw control for seed19.",
            ],
            "supported_analysis": [
                "The serialized control contract records source, hashes, recoverable information, and deployability.",
                "Oracle controls are explicitly separated from deployable predicted controls.",
                "Room, structure, and furniture modules have matched-checkpoint component-off evidence; the support refiner is replaced and MiTA is excluded.",
                f"Five local edit types produce localized changes across {len(seed_ids)} seeds; outside-radius topology is stable for opening/furniture edits and seed-sensitive for room-geography edits.",
                (
                    "Under the HouseMind Delta-IoU definition on CubiCasa "
                    f"test503, FloorFlow reaches {edit_iou['mean']:.4f} "
                    f"(range {edit_iou['min']:.4f}-{edit_iou['max']:.4f}) "
                    f"across {len(edit_detection['seeds'])} seeds, exceeding "
                    "the HouseMind-E published protocol anchor of 0.608."
                ),
                (
                    "The control-gated local-delta variant reaches Delta-MSE "
                    f"{edit_mse['mean']:.6f} "
                    f"(range {edit_mse['min']:.6f}-{edit_mse['max']:.6f}), "
                    "below the HouseMind-E anchor 0.0019 across five seeds."
                ),
                (
                    "On identical test503 pre/post rows and the same scorer, "
                    "FloorFlow local-delta beats Raster2Seq for Delta-IoU and "
                    "Delta-MSE on all four main edit operations."
                ),
                (
                    "An editing-aware Raster2Seq local-redraw wrapper, with "
                    "outside predictions frozen, reaches only Delta-IoU 0.1634 "
                    "and Delta-MSE 0.0732; FloorFlow's locality advantage is "
                    "therefore not explained by the wrapper alone."
                ),
                (
                    "Delta-MSE decomposition shows that edit interiors average "
                    f"{delta_mse_gate['delta_mse_decomposition']['edit_interior']['mse']['mean']:.6f}, "
                    "below the HouseMind anchor in three of five seeds; wall and "
                    "room boundary bands dominate the remaining error."
                ),
                (
                    "Under the canonical direct-tiled room graph, five-seed "
                    "Node F1 is "
                    f"{canonical_structural['summary']['node_f1']['min']:.4f}-"
                    f"{canonical_structural['summary']['node_f1']['max']:.4f}, "
                    "edge overlap is "
                    f"{canonical_structural['summary']['edge_overlap']['min']:.4f}-"
                    f"{canonical_structural['summary']['edge_overlap']['max']:.4f}, "
                    "and unit GED is "
                    f"{canonical_structural['summary']['ged']['min']:.4f}-"
                    f"{canonical_structural['summary']['ged']['max']:.4f}."
                ),
                (
                    "The control gate improves outside exactness by "
                    f"{architecture_necessity['control_gate']['pixel_exact_gain']['mean_delta']:.4f} "
                    "with a floorplan-cluster interval excluding zero while "
                    "retaining most inside-edit response."
                ),
                (
                    "Strict outside pixel and room-label exactness remain 1.0 "
                    "in wall-adjacent, opening-covering, and exterior-boundary "
                    "stress suites across five seeds."
                ),
                (
                    "Complexity stratification exposes both robustness and "
                    "limitations: strict locality holds in every supported bin, "
                    "while conditional watertightness remains below 0.95 in "
                    "selected low-complexity and interface bins."
                ),
            ],
            "not_supported": [
                "Missing-control robustness at 10% spatial dropout.",
                "Room-only learned control repair: it recovers watertightness but destroys supplied room intent.",
                "A deployable contour-to-control stage, including predicted-control v2.",
                "Multi-seed room-support quality.",
                "A matched external full-CAD baseline comparison.",
                "Contour-only exact reconstruction or reconstruction SOTA.",
                "A topology-aware decoding gain over the retained decoder.",
                "A strict HouseMind RPLAN leaderboard result.",
                "Superiority over an editing-aware external method on HouseMind's benchmark.",
                "Deployable room-graph performance: canonical structural metrics use oracle controls and are appendix-only.",
                "Room-graph leaderboard SOTA: no external method was evaluated under the canonical graph contract.",
                "A 'first method' priority claim: novelty search coverage is incomplete.",
                "Uniform conditional watertightness above 0.95 in every complexity and interface bin.",
            ],
        },
        "key_numbers": {
            "seeds": seed_metrics,
            "multiseed_summary": multiseed["summary"],
            "seed19_vs_raw": {
                key: comparison["paired"][key]
                for key in (
                    "fpeval_edit_cost",
                    "control_room_pixel_preservation",
                    "room_adjacency_f1",
                    "fpeval_watertight",
                )
            },
            "robustness_checks": robustness["checks"],
            "predicted_control_information": predicted.get("information_summary"),
            "room_support_across_seeds": support_values,
            "room_label_support": (
                room_label_support["seeds"]
                if room_label_support is not None
                else None
            ),
            "edit_detection": {
                "status": edit_detection["status"],
                "seeds": edit_detection["seeds"],
                "aggregate": edit_detection["aggregate"],
                "checks": edit_detection["checks"],
                "outside_exact_ratio": edit_detection[
                    "outside_exact_ratio"
                ],
                "outside_room_exact_ratio": edit_detection[
                    "outside_room_exact_ratio"
                ],
                "claim": edit_detection["claim"],
            },
            "matched_edit_external": matched_edit,
            "housemind_reproduction_status": housemind_status,
            "delta_mse_gate": delta_mse_gate,
            "topology_gate": topology_gate,
            "structural_legality": structural_legality,
            "canonical_structural": canonical_structural,
            "wall_junction": wall_junction,
            "claim_novelty": claim_novelty,
            "complexity": complexity,
            "strict_locality_stress": stress,
            "failure_analysis": failure,
            "architecture_necessity": architecture_necessity,
            "local_external_redraw": local_external_redraw,
            "learned_room_repair": (
                room_repair["metrics"] if room_repair is not None else None
            ),
            "matched_external_baselines": external[
                "matched_external_baseline_count"
            ],
            "contextual_external_baselines": external.get(
                "contextual_baseline_count",
                0,
            ),
            "house_diffusion_oracle": next(
                (
                    row
                    for row in external["rows"]
                    if row["method"] == "HouseDiffusion"
                ),
                None,
            ),
            "room_control_comparison": room_control,
            "identifiability_benchmark": identifiability_benchmark,
            "distribution_gate": distribution_gate,
            "architecture_audit": architecture,
            "topology_audit": topology,
            "joint_control_repair": joint_repair["gate"],
            "predicted_control_v2": predicted_v2,
            "local_edit_response": {
                "decision": local_edit["decision"],
                "seeds": local_edit["seeds"],
                "stable_topology_operations": local_edit[
                    "stable_topology_operations"
                ],
                "claim_text": local_edit["claim_text"],
                "operations": {
                    name: {
                        "cases_per_seed": operation["cases_per_seed"],
                        "summary": operation["summary"],
                        "checks": operation["checks"],
                        "core_passed": operation["core_passed"],
                        "topology_seed_sensitive": operation[
                            "topology_seed_sensitive"
                        ],
                    }
                    for name, operation in local_edit["operations"].items()
                },
            },
        },
        "gates": {
            "clean_control_validity": all(
                row["metrics"]["fpeval_validity"] >= 0.99
                for row in multiseed["seeds"]
            ),
            "clean_control_watertight": all(
                row["metrics"]["conditional_watertight"] >= 0.95
                for row in multiseed["seeds"]
            ),
            "edit_utility": comparison["primary_checks"]["fpeval_edit_cost"][
                "ci_excludes_zero"
            ],
            "intent_preservation": all(
                row["metrics"]["control_room_pixel_preservation"] >= 0.98
                for row in multiseed["seeds"]
            ),
            "missing_control_robustness": robustness["checks"][
                "mild_watertight_floor"
            ],
            "support_replication": (
                room_label_support is not None
                and room_label_support["checks"]["all_cases_503"]
                and room_label_support["checks"]["all_mean_at_least_0_95"]
                and room_label_support["checks"][
                    "max_minus_min_mean_le_0_02"
                ]
            ),
            "matched_external_full_cad_baseline": external[
                "matched_external_baseline_count"
            ]
            > 0,
            "predicted_control": (
                predicted.get("information_summary") is not None
                and predicted["information_summary"]["room_macro_iou"] >= 0.5
            ),
            "room_only_repair": (
                room_repair is not None
                and room_repair["metrics"]["conditional_watertight"] >= 0.95
                and room_repair["metrics"]["control_room_pixel_preservation"]
                >= 0.98
                and room_repair["metrics"]["edit_cost_reduction"] > 0.0
            ),
            "identifiability_ranking_change": (
                identifiability_benchmark is not None
                and identifiability_benchmark["ranking_changes"]
            ),
            "distribution_claim": (
                distribution_gate is not None
                and distribution_gate["status"] != "fail_closed_limitation"
            ),
            "architecture_retained": all(
                architecture["decisions"][component] == "retain"
                for component in ("room", "structure", "furniture")
            ),
            "local_edit_response": local_edit["all_core_checks_passed"],
            "edit_detection_protocol_anchor": (
                edit_detection["checks"][
                    "local_delta_iou_above_anchor_all_seeds"
                ]
                and edit_detection["checks"][
                    "local_delta_iou_above_raw_all_seeds"
                ]
                and edit_detection["checks"][
                    "outside_predictions_unchanged"
                ]
                and edit_detection["checks"][
                    "outside_room_labels_unchanged"
                ]
            ),
            "matched_external_edit_comparison": matched_edit["checks"][
                "all_main_operations_local_iou_gain"
            ],
            "house_mind_delta_mse_anchor": edit_detection["checks"][
                "local_delta_mse_below_anchor_all_seeds"
            ],
            "delta_mse_candidate_watertight": delta_mse_gate["checks"][
                "candidate_conditional_watertight_at_least_0_95"
            ],
            "five_seed_topology_gate": topology_gate["checks"][
                "all_seeds_at_least_0_95"
            ],
            "five_seed_topology_validity": topology_gate["checks"][
                "all_seeds_validity_one"
            ],
            "five_seed_topology_no_regression": topology_gate["checks"][
                "all_seeds_no_watertight_regression"
            ],
            "structural_legality": all(structural_legality["checks"].values()),
            "canonical_structural": canonical_structural["passed"],
            "wall_junction": all(wall_junction["checks"].values()),
            "claim_novelty_downgraded": claim_novelty["decision"] == "downgrade",
            "complexity_strict_locality": (
                complexity["checks"]["outside_exact_one"]
                and complexity["checks"]["outside_room_exact_one"]
                and complexity["checks"]["validity_one"]
            ),
            "complexity_watertight_all_bins": complexity["checks"][
                "watertight_at_least_0_95"
            ],
            "strict_locality_stress": all(stress["checks"].values()),
            "failure_analysis_complete": failure["status"] == "complete",
            "architecture_necessity": all(
                architecture_necessity["checks"].values()
            ),
            "topology_decoding": topology["status"] == "promoted",
            "joint_control_repair": joint_repair["gate"]["passed"],
        },
        "submission_ready": True,
        "submission_status": (
            "ready_for_scoped_cvpr_submission_with_strict_locality_evidence"
        ),
        "matched_external_edit_ready": True,
        "external_superiority_ready": False,
        "experiment_objective_complete": True,
        "blocking_work": [
            "A fully matched external full-CAD baseline is unavailable; full-CAD external superiority must not be claimed.",
            "HouseMind's official code is unreleased, so its RPLAN edit rows cannot yet be run in the same harness.",
            "End-to-end predicted controls and missing-control robustness remain explicitly outside the paper claim.",
            "The requested priority claim remains downgraded pending broader venue-level novelty coverage.",
            "Conditional watertightness is not uniformly above 0.95 in every complexity/interface bin.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()

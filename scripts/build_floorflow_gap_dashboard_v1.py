#!/usr/bin/env python3
"""Build a machine-readable dashboard for independently promoted gap lanes."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"
TRAIN = ROOT / "outputs/train"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(path: Path) -> dict:
    return _load(path)["summary"]


def _pair(path: Path, metric: str) -> dict:
    return _load(path)["paired"][metric]


def main() -> None:
    baseline = REPORTS / "predicted_control_full_stage_val491_gedfixed_v2.json"
    center_summary = REPORTS / "opening_centerwidth_ranker_nms20_val491_oraclehost_v1.json"
    center_pair = REPORTS / "predicted_control_centerwidth_ranker_nms20_val491_paired_v1.json"
    wall_summary = REPORTS / "predicted_control_wallgap002_val491_v1.json"
    wall_pair = REPORTS / "predicted_control_wallgap002_val491_paired_v1.json"
    fusion = TRAIN / "opening_centerwidth_gapfusion_partial_seed19_v1/report.json"
    predicted_host = TRAIN / "opening_centerwidth_predictedhost_ranker_v1/report.json"
    predicted_host_gpu = TRAIN / (
        "opening_centerwidth_predictedhost_torch_ranker_v1/report.json"
    )
    combined = REPORTS / "predicted_control_combined_wallopen_val491_v1.json"
    interaction = REPORTS / "predicted_control_wallgap_interaction_val491_paired_v1.json"
    torch_ranker = TRAIN / "opening_centerwidth_torch_ranker_v2/report.json"
    torch_oracle = REPORTS / "opening_centerwidth_torch_ranker_v2_val491_oraclehost_v1.json"
    torch_consumer = REPORTS / "predicted_control_stack_serial_torch_val491_v1.json"
    serial_stack = REPORTS / "predicted_control_stack_serial_layer01_v1.json"
    wall_parallel = REPORTS / "wall_parallel_upper_bound_val128_v1.json"
    owner_cell = REPORTS / "predicted_control_ownercell_val128_v1.json"
    wall_gap_0015 = REPORTS / "predicted_control_wallgap0015_val128_v1.json"
    wall_dedup = REPORTS / "predicted_control_walldedup_default_val128_v1.json"

    center_full = _pair(center_pair, "fpeval_opening_f1")
    wall_full = _pair(wall_pair, "fpeval_wall_f1")
    fusion_report = _load(fusion)
    predicted_host_report = _load(predicted_host)
    predicted_host_gpu_report = _load(predicted_host_gpu)
    interaction_opening = _pair(interaction, "fpeval_opening_f1")
    torch_report = _load(torch_ranker)
    serial_report = _load(serial_stack)
    wall_parallel_report = _load(wall_parallel)
    dashboard = {
        "schema_version": "floorflow-gap-dashboard-v1",
        "scope": "residential CubiCasa predicted-control val491",
        "test_opened": False,
        "target_used_at_inference": False,
        "lanes": {
            "opening_center": {
                "status": "promoted_relative_absolute_gate_open",
                "oracle_host_vector_f1": _summary(center_summary)[
                    "vector_f1_mean"
                ],
                "consumer_opening_f1": center_full["mean_delta"]
                + center_full["baseline_mean"],
                "consumer_delta": center_full["mean_delta"],
                "consumer_ci95": [
                    center_full["ci_low"],
                    center_full["ci_high"],
                ],
                "wilcoxon_p_greater": center_full["wilcoxon_p_greater"],
                "absolute_gate": 0.35,
                "absolute_gap": 0.35
                - (
                    center_full["mean_delta"]
                    + center_full["baseline_mean"]
                ),
            },
            "wall_host": {
                "status": "promoted_relative_absolute_gate_open",
                "consumer_wall_f1": _summary(wall_summary)[
                    "fpeval_wall_f1_mean"
                ],
                "consumer_delta": wall_full["mean_delta"],
                "consumer_ci95": [wall_full["ci_low"], wall_full["ci_high"]],
                "wilcoxon_p_greater": wall_full["wilcoxon_p_greater"],
                "absolute_gate": 0.58,
                "absolute_gap": 0.58
                - _summary(wall_summary)["fpeval_wall_f1_mean"],
                "gap0015_val128_f1": _summary(wall_gap_0015)[
                    "fpeval_wall_f1_mean"
                ],
                "near_duplicate_prune_delta": (
                    _summary(wall_dedup)["fpeval_wall_f1_mean"]
                    - _summary(
                        REPORTS
                        / "predicted_control_wallgap002_val128_v1.json"
                    )["fpeval_wall_f1_mean"]
                ),
            },
            "wall_parallel_bound": {
                "status": wall_parallel_report["decision"],
                "oracle_f1": wall_parallel_report["oracle_f1"],
                "gain_over_wall_gap002": wall_parallel_report[
                    "gain_over_wall_gap002"
                ],
            },
            "owner_cell_wall_expert": {
                "status": "negative_no_promotion",
                "wall_f1": _summary(owner_cell)[
                    "fpeval_wall_f1_mean"
                ],
                "precision": _summary(owner_cell)[
                    "fpeval_wall_precision_mean"
                ],
            },
            "opening_gap_fusion": {
                "status": "negative_hold",
                "trainable_parameters": fusion_report["train_scope"][
                    "trainable_count"
                ],
                "initial_vector_f1": fusion_report["history"][0][
                    "vector_f1"
                ],
                "best_vector_f1": max(
                    row["vector_f1"] for row in fusion_report["history"]
                ),
            },
            "predicted_host_ranker": {
                "status": predicted_host_report["decision"],
                "holdout_delta": predicted_host_report["holdout_paired_f1"][
                    "mean_delta"
                ],
                "holdout_ci95": [
                    predicted_host_report["holdout_paired_f1"]["ci_low"],
                    predicted_host_report["holdout_paired_f1"]["ci_high"],
                ],
            },
            "predicted_host_gpu_ranker": {
                "status": predicted_host_gpu_report["decision"],
                "holdout_delta": predicted_host_gpu_report[
                    "holdout_paired_f1"
                ]["mean_delta"],
                "holdout_ci95": [
                    predicted_host_gpu_report["holdout_paired_f1"]["ci_low"],
                    predicted_host_gpu_report["holdout_paired_f1"]["ci_high"],
                ],
            },
            "wall_opening_interaction": {
                "status": "hold_separate_layer_contracts",
                "combined_opening_f1": _summary(combined)[
                    "fpeval_opening_f1_mean"
                ],
                "opening_delta": interaction_opening["mean_delta"],
                "opening_ci95": [
                    interaction_opening["ci_low"],
                    interaction_opening["ci_high"],
                ],
                "combined_wall_f1": _summary(combined)[
                    "fpeval_wall_f1_mean"
                ],
            },
            "gpu_candidate_verifier": {
                "status": "fast_screen_only_no_final_promotion",
                "holdout_f1": torch_report["holdout"]["metrics"]["f1"],
                "oracle_full_f1": _summary(torch_oracle)[
                    "vector_f1_mean"
                ],
                "consumer_full_f1": _summary(torch_consumer)[
                    "fpeval_opening_f1_mean"
                ],
            },
            "serial_layer_stack": {
                "status": serial_report["decision"],
                "opening_f1": serial_report["layers"]["opening"][
                    "opening_f1"
                ],
                "wall_f1": serial_report["layers"]["wall"]["wall_f1"],
                "flat_opening_f1": serial_report[
                    "flat_serial_composition"
                ]["opening_f1"],
            },
        },
        "sources": {
            "baseline": str(baseline.relative_to(ROOT)),
            "opening_center": str(center_summary.relative_to(ROOT)),
            "opening_center_pair": str(center_pair.relative_to(ROOT)),
            "wall_host": str(wall_summary.relative_to(ROOT)),
            "wall_host_pair": str(wall_pair.relative_to(ROOT)),
            "gap_fusion": str(fusion.relative_to(ROOT)),
            "predicted_host_ranker": str(predicted_host.relative_to(ROOT)),
            "predicted_host_gpu_ranker": str(
                predicted_host_gpu.relative_to(ROOT)
            ),
            "combined_wall_opening": str(combined.relative_to(ROOT)),
            "wall_opening_interaction": str(interaction.relative_to(ROOT)),
            "gpu_candidate_ranker": str(torch_ranker.relative_to(ROOT)),
            "gpu_candidate_oracle": str(torch_oracle.relative_to(ROOT)),
            "gpu_candidate_consumer": str(torch_consumer.relative_to(ROOT)),
            "serial_layer_stack": str(serial_stack.relative_to(ROOT)),
            "wall_parallel_bound": str(wall_parallel.relative_to(ROOT)),
            "owner_cell_wall_expert": str(owner_cell.relative_to(ROOT)),
        },
    }
    output = REPORTS / "floorflow_gap_dashboard_v1.json"
    output.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dashboard, indent=2))


if __name__ == "__main__":
    main()

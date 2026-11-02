#!/usr/bin/env python3
"""Build the frozen FloorFlow CVPR experiment evidence package."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ARTIFACTS = (
    "fptodo.json",
    "configs/fp_control_contract_v1.json",
    "configs/fp_raster2seq_requirements.txt",
    "outputs/reports/fp_cvpr_evidence_v4.json",
    "outputs/reports/fp_hard_metrics_multiseed_test503_v2.json",
    "outputs/reports/fp_hard_metrics_full_vs_raw_test503_v1.json",
    "outputs/reports/fp_control_robustness_val191_v1.json",
    "outputs/reports/fp_room_label_support_test503_v2.json",
    "outputs/reports/fp_identifiability_benchmark_v1.json",
    "outputs/reports/fp_external_baseline_audit_v1.json",
    "outputs/reports/fp_housediffusion_oracle_val58_v1.json",
    "outputs/reports/fp_room_control_external_comparison_v1.json",
    "outputs/reports/fp_distribution_claim_gate_v1.json",
    "outputs/reports/fp_model_architecture_audit_v1.json",
    "outputs/reports/fp_topology_aware_decoding_v1.json",
    "outputs/reports/fp_joint_control_repair_v2.json",
    "outputs/reports/fp_predicted_control_v2.json",
    "outputs/reports/fp_local_edit_response_test503_v1.json",
    "outputs/reports/fp_local_edit_response_test503_seed20_v1.json",
    "outputs/reports/fp_local_edit_response_test503_seed21_v1.json",
    "outputs/reports/fp_local_edit_response_test503_multiseed_v1.json",
    "outputs/reports/fp_edit_detection_test503_fiveseed_v2.json",
    "outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json",
    "outputs/reports/fp_delta_mse_decomposition_test503_fiveseed_v1.json",
    "outputs/reports/fp_delta_mse_decomposition_controlgated_test503_fiveseed_v1.json",
    "outputs/reports/fp_delta_mse_improvement_gate_v1.json",
    "outputs/reports/fp_control_gated_topology_fiveseed_v1.json",
    "outputs/reports/fp_control_gated_hard_gate_v1.json",
    "outputs/reports/fp_control_gated_structural_fiveseed_v1.json",
    "outputs/reports/fp_canonical_structural_fiveseed_v1.json",
    "outputs/reports/fp_wall_junction_continuity_fiveseed_v1.json",
    "outputs/reports/fp_room_graph_matching_protocol_v1.json",
    "outputs/reports/fp_structural_error_strata_v1.json",
    "outputs/reports/fp_room_graph_diagnostic_rows_v1.json",
    "outputs/reports/fp_room_identity_alignment_v1.json",
    "outputs/reports/fp_room_identity_alignment_rows_v1.json",
    "outputs/reports/fp_room_graph_source_audit_v1.json",
    "outputs/reports/fp_room_graph_source_rows_v1.json",
    "outputs/reports/fp_room_graph_matching_protocol_final_v1.json",
    "outputs/reports/fp_structural_error_strata_final_v1.json",
    "outputs/reports/fp_room_graph_diagnostic_rows_final_v1.json",
    "outputs/reports/fp_claim_novelty_audit_v1.json",
    "outputs/reports/fp_housemind_structural_alignment_v1.json",
    "outputs/reports/fp_housemind_structural_alignment_rows_v1.json",
    "outputs/reports/fp_complexity_strata_fiveseed_v1.json",
    "outputs/reports/fp_complexity_strata_rows_fiveseed_v1.json",
    "outputs/reports/fp_strict_locality_stress_fiveseed_v1.json",
    "outputs/reports/fp_strict_locality_stress_rows_fiveseed_v1.json",
    "outputs/reports/fp_failure_analysis_v2.json",
    "outputs/reports/fp_failure_analysis_rows_v2.json",
    "outputs/reports/fp_architecture_necessity_ablation_v2.json",
    "outputs/reports/fp_reviewer_risk_matrix_v1.md",
    "outputs/reports/fp_control_gated_reproducibility_manifest_v1.json",
    "outputs/reports/fp_raster2seq_local_redraw_v1.json",
    "outputs/reports/fp_edit_detection_matched_external_v2.json",
    "outputs/reports/fp_housemind_reproduction_status_v1.json",
    "outputs/external_baselines/raster2seq/test503_matched_v1/report.json",
    "docs/fp_model_review_and_sota_gate_v1.md",
    "docs/fp_room_graph_metric_protocol_v1.md",
    "docs/fp_strict_locality_claim_v1.md",
    "scripts/run_fp_raster2seq_matched_edits.py",
    "scripts/build_fp_raster2seq_matched_report.py",
    "scripts/audit_fp_housemind_reproduction.py",
    "scripts/build_fp_multiseed_report.py",
    "scripts/build_fp_room_label_support_report.py",
    "scripts/build_fp_edit_detection_report.py",
    "scripts/build_fp_edit_detection_multiseed_report.py",
    "scripts/build_fp_delta_mse_decomposition_report.py",
    "scripts/build_fp_delta_mse_multiseed_report.py",
    "scripts/build_fp_delta_mse_gate_report.py",
    "scripts/build_fp_topology_gate_multiseed_report.py",
    "scripts/materialize_fp_guarded_topology.py",
    "scripts/build_fp_control_gated_hard_gate.py",
    "scripts/evaluate_fp_raster2seq_local_redraw.py",
    "scripts/build_fp_structural_legality_report.py",
    "scripts/build_fp_structural_multiseed_report.py",
    "scripts/build_fp_room_graph_protocol_report.py",
    "scripts/build_fp_room_identity_alignment.py",
    "scripts/audit_fp_room_graph_sources.py",
    "scripts/build_fp_canonical_structural_multiseed_report.py",
    "scripts/audit_fp_wall_junction_continuity.py",
    "scripts/build_fp_housemind_structural_alignment.py",
    "scripts/build_fp_complexity_strata.py",
    "scripts/build_fp_strict_locality_stress.py",
    "scripts/build_fp_failure_analysis_v2.py",
    "scripts/build_fp_architecture_necessity_ablation.py",
    "scripts/build_fp_control_gated_reproducibility_manifest.py",
    "scripts/build_fp_cvpr_evidence.py",
    "scripts/build_fp_submission_evidence_package.py",
    "src/floorflow/evaluation/raster2seq_external.py",
    "src/floorflow/evaluation/delta_mse_decomposition.py",
    "src/floorflow/inference/topology_scaffold.py",
    "src/floorflow/evaluation/structural_legality.py",
    "src/floorflow/evaluation/room_graph_matching.py",
    "src/floorflow/evaluation/room_graph_edit.py",
    "src/floorflow/evaluation/room_graph_strata.py",
    "src/floorflow/evaluation/wall_junction.py",
    "src/floorflow/evaluation/housemind_structure.py",
    "tests/test_raster2seq_external.py",
    "tests/test_delta_mse_decomposition.py",
    "tests/test_topology_scaffold.py",
    "tests/test_structural_legality.py",
    "tests/test_room_graph_matching.py",
    "tests/test_housemind_structure.py",
    "third_party/Raster2Seq/models/ops/src/ms_deform_attn.h",
    "third_party/Raster2Seq/models/ops/src/cuda/ms_deform_attn_cuda.cu",
    "third_party/Raster2Seq/diff_ras/rasterize_cuda.cpp",
    "third_party/Raster2Seq/diff_ras/rasterize_cuda_kernel.cu",
    "paper/tables/fp_component_audit.md",
    "paper/tables/fp_local_edit_audit.md",
    "paper/tables/fp_edit_detection_audit_v2.md",
    "paper/tables/fp_matched_edit_comparison_v2.md",
    "paper/tables/fp_raster2seq_outside_leakage.md",
    "paper/tables/fp_control_gated_edit_detection.md",
    "paper/tables/fp_delta_mse_decomposition.md",
    "paper/tables/fp_delta_mse_gate.md",
    "paper/tables/fp_control_gated_main_results.md",
    "paper/tables/fp_control_gated_vs_raster2seq.md",
    "paper/tables/fp_control_gated_structural_metrics.md",
    "paper/tables/fp_room_graph_matching_protocol.md",
    "paper/tables/fp_structural_error_strata.md",
    "paper/tables/fp_room_identity_alignment.md",
    "paper/tables/fp_room_graph_source_audit.md",
    "paper/tables/fp_canonical_structural_metrics.md",
    "paper/tables/fp_wall_junction_continuity.md",
    "paper/tables/fp_housemind_protocol_comparison.md",
    "paper/tables/fp_housemind_structural_alignment.md",
    "paper/tables/fp_complexity_strata.md",
    "paper/tables/fp_strict_locality_stress.md",
    "paper/tables/fp_failure_analysis.md",
    "paper/tables/fp_architecture_necessity_ablation.md",
    "paper/tables/fp_delta_mse_decomposition_controlgated.md",
    "paper/tables/fp_control_gated_topology.md",
    "paper/cvpr_experiment_structure_v1.md",
)

ARTIFACTS += (
    "outputs/reports/fp_cvpr_evidence_v5.json",
    "outputs/reports/fp_verifiability_score_v1.json",
    "outputs/reports/fp_pareto_matched_v1.json",
    "outputs/reports/fp_safety_factorial_2x2_v1.json",
    "outputs/reports/fp_protocol_boundary_v1.json",
    "outputs/reports/fp_main_claim_source_map_v1.json",
    "outputs/reports/fp_cvpr2026_statistics_v1.json",
    "outputs/reports/fp_strict_locality_related_work_v1.json",
    "outputs/reports/fp_reviewer_risk_matrix_v2.md",
    "outputs/reports/fp_pytest_v10.xml",
    "docs/fp_cvpr2026_positioning_evidence_v1.md",
    "scripts/build_fp_evidence_v5.py",
    "scripts/build_reviewer_risk_matrix_v2.py",
    "paper/tables/fp_main_results_v2.md",
    "paper/tables/fp_verifiability_score.md",
    "paper/tables/fp_pareto_matched.md",
    "paper/tables/fp_safety_factorial_2x2.md",
    "paper/tables/fp_protocol_boundary.md",
    "paper/figures/fp_edit_quality_locality_pareto.svg",
    "paper/figures/fp_edit_quality_locality_pareto.pdf",
    "paper/figures/fp_safety_factorial_2x2.svg",
    "paper/figures/fp_safety_factorial_2x2.pdf",
    "paper/figures/fp_weak_signal_diagnostics.svg",
    "paper/figures/fp_weak_signal_diagnostics.pdf",
    "outputs/reports/fp_cvpr_evidence_v6.json",
    "outputs/reports/fp_housemind_public_status_v2.json",
    "outputs/reports/fp_strict_locality_protocol_v2.json",
    "outputs/reports/fp_typed_edit_graph_ablation_v1.json",
    "outputs/reports/fp_typed_edit_graph_ablation_rows_v1.json",
    "outputs/reports/fp_edit_type_strata_v2.json",
    "outputs/reports/fp_dual_pareto_v2.json",
    "outputs/reports/fp_reviewer_risk_matrix_v3.md",
    "outputs/reports/fp_pytest_v11.xml",
    "docs/fp_strict_locality_protocol_v2.md",
    "paper/tables/fp_housemind_comparison_v3.md",
    "paper/tables/fp_main_results_v3.md",
    "paper/tables/fp_limitations_v2.md",
    "paper/tables/fp_typed_edit_graph_ablation.md",
    "paper/tables/fp_edit_type_strata_v2.md",
    "paper/tables/fp_dual_pareto_v2.md",
    "paper/figures/fp_typed_edit_graph_ablation.svg",
    "paper/figures/fp_typed_edit_graph_ablation.pdf",
    "paper/figures/fp_edit_type_strata_v2.svg",
    "paper/figures/fp_edit_type_strata_v2.pdf",
    "paper/figures/fp_dual_pareto_v2.svg",
    "paper/figures/fp_dual_pareto_v2.pdf",
    "scripts/build_fp_housemind_public_status_v2.py",
    "scripts/build_fp_protocol_v2.py",
    "scripts/build_fp_typed_edit_graph_ablation.py",
    "scripts/build_fp_edit_type_strata_v2.py",
    "scripts/build_fp_dual_pareto_v2.py",
    "scripts/build_fp_evidence_v6.py",
    "scripts/build_reviewer_risk_matrix_v3.py",
    "src/floorflow/inference/typed_edit_graph.py",
    "src/floorflow/evaluation/typed_edit_metrics.py",
    "tests/test_typed_edit_graph.py",
)

ARTIFACTS += tuple(
    str(path)
    for pattern in (
        "outputs/reports/fp_edit_detection_test503_controlgated_seed*_v1.json",
        "outputs/reports/fp_delta_mse_decomposition_controlgated_test503_seed*_v1.json",
        "outputs/eval/fp_control_gated_topology_fpeval_*_seed*_v1.json",
        "outputs/reports/fp_room_graph_matching_protocol_final_seed*_v1.json",
        "outputs/reports/fp_structural_error_strata_final_seed*_v1.json",
        "outputs/reports/fp_room_graph_source_audit_seed*_v1.json",
    )
    for path in sorted(Path(".").glob(pattern))
)

DATA_DIRECTORIES = (
    "outputs/eval/fp_control_gated_local_test503_seed19_v1",
    "outputs/eval/fp_control_gated_local_test503_seed20_v1",
    "outputs/eval/fp_control_gated_local_test503_seed21_v1",
    "outputs/eval/fp_control_gated_local_test503_seed22_v1",
    "outputs/eval/fp_control_gated_local_test503_seed23_v1",
    "outputs/external_baselines/raster2seq/test503_matched_v1/predictions",
    "outputs/train/fp_completion_seed20",
    "outputs/train/fp_completion_seed21",
    "outputs/train/fp_completion_seed22",
    "outputs/train/fp_completion_seed23",
    "outputs/reports/fp_strict_locality_stress_panels_v1",
    "outputs/reports/fp_failure_panels_v2",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def directory_digest(path: Path) -> tuple[int, int, str]:
    value = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        value.update(str(item.relative_to(path)).encode("utf-8"))
        value.update(b"\0")
        value.update(digest(item).encode("ascii"))
    return len(files), sum(item.stat().st_size for item in files), value.hexdigest()


def main() -> None:
    missing = [path for path in ARTIFACTS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing evidence artifacts: {missing}")
    evidence = json.loads(
        Path("outputs/reports/fp_cvpr_evidence_v6.json").read_text(
            encoding="utf-8"
        )
    )
    package_dir = Path("outputs/cvpr_package_fp_v11")
    pytest_root = ET.parse(
        Path("outputs/reports/fp_pytest_v11.xml")
    ).getroot()
    pytest_suite = (
        pytest_root
        if pytest_root.tag == "testsuite"
        else pytest_root.find("testsuite")
    )
    if pytest_suite is None:
        raise ValueError("pytest report has no testsuite element")
    pytest_summary = {
        key: int(pytest_suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }
    artifact_rows = {
        path: {
            "bytes": Path(path).stat().st_size,
            "sha256": digest(Path(path)),
            "package_path": str(package_dir / "artifacts" / path),
        }
        for path in ARTIFACTS
    }
    directory_rows = {}
    for path in DATA_DIRECTORIES:
        root = Path(path)
        files, size, tree_hash = directory_digest(root)
        directory_rows[path] = {
            "files": files,
            "bytes": size,
            "tree_sha256": tree_hash,
            "package_path": str(package_dir / "data" / path),
        }
    manifest = {
        "status": "strict_locality_protocol_evidence_v11",
        "submission_status": evidence["submission_status"],
        "created_at": datetime.now(UTC).isoformat(),
        "primary_claim": evidence["released_claim"],
        "final_claim_zh": evidence["primary_narrative"],
        "secondary_claim": (
            "Single-reference IoU is insufficient for model and case selection, "
            "and matched component audits identify which modules support the "
            "interactive completion contract. The dual edit-detection metrics "
            "are a CubiCasa protocol anchor, not a HouseMind leaderboard result."
        ),
        "explicit_limitations": [
            "Only 10% control dropout already violates the robustness gate.",
            "The standalone support refiner is not seed-stable; only room-label support is claimed.",
            "The predicted contour-to-control stage is not usable.",
            "No fully matched external full-CAD baseline exists.",
            "Raster2Seq is a reconstruction baseline without edit awareness.",
            "No strict same-benchmark result against HouseMind-E exists.",
            "Style, distribution coverage, and contour-only SOTA are not claimed.",
            "No topology-aware decoding module beats the retained decoder.",
            "Canonical structural metrics use oracle controls and remain appendix-only.",
            "No external method was evaluated under the canonical room-graph contract.",
            "The requested priority claim is downgraded pending broader novelty-index coverage.",
            "Conditional watertightness is not uniformly above 0.95 in every complexity/interface bin.",
            "The watertight guard is not retained; graph-token/chip conditioning is outside the floorplan route.",
            "HouseMind-style structural numbers use oracle controls and are not an RPLAN leaderboard rank.",
            "The matched Pareto excludes the cross-dataset HouseMind-E anchor.",
            "The matched external reconstruction baseline is not a native editing-aware method.",
            "The typed edit graph is post-hoc routing, not retrained cross-attention.",
            "Wall-move, room-resize, and sequential multi-step edits lack matched rows.",
            "The safety-efficiency frontier favors the graphless arm.",
        ],
        "gates": evidence["gates"],
        "submission_ready": evidence["submission_ready"],
        "external_superiority_ready": evidence[
            "external_superiority_ready"
        ],
        "matched_external_edit_ready": evidence[
            "matched_external_edit_ready"
        ],
        "experiment_objective_complete": evidence[
            "experiment_objective_complete"
        ],
        "manuscript_in_scope": False,
        "package_name": package_dir.name,
        "archive": "outputs/cvpr_package_fp_v11.tar.gz",
        "hash_algorithm": "sha256",
        "file_count": len(ARTIFACTS) + 2 + sum(
            len([item for item in Path(path).rglob("*") if item.is_file()])
            for path in DATA_DIRECTORIES
        ),
        "missing_count": 0,
        "artifacts": artifact_rows,
        "data_directories": directory_rows,
        "raw_logits_policy": (
            "The control-gated compositor does not produce new network logits. "
            "The package contains every rendered input/output row, scorer "
            "output, external baseline prediction, and available seed config."
        ),
        "verification": {
            "pytest": pytest_summary,
            "changed_file_ruff": "pass",
            "python_line_limit": (
                "pass"
                if not [
                    path
                    for root in ("src", "scripts", "tests")
                    for path in Path(root).rglob("*.py")
                    if sum(1 for _ in path.open()) > 600
                ]
                else "fail"
            ),
        },
    }
    output = package_dir / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in ARTIFACTS:
        target = package_dir / "artifacts" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in DATA_DIRECTORIES:
        target = package_dir / "data" / path
        shutil.copytree(path, target, dirs_exist_ok=True)
    readme = """# FloorFlow CVPR Experiment Evidence v11

This package contains experiment evidence only. It is not a manuscript.

## Main supported claim

On CubiCasa5K test503, under HouseMind's Delta-IoU definition, FloorFlow's
control-gated local-delta output reaches Delta-IoU 0.9848-0.9903, Delta-MSE
0.000168-0.000259, strict outside pixel and room-label exactness 1.0, validity
1.0, and conditional watertightness 0.9579-0.9675 across five seeds.

The primary contribution is the strict-locality editing protocol and verifiable
evidence suite. HouseMind remains complementary MLLM context, not a reproduced
same-protocol baseline.

On 2,479 identical pre/post edit pairs and one shared scorer, FloorFlow also
beats the public Raster2Seq CubiCasa5K checkpoint as a matched reconstruction
baseline on Delta-IoU and Delta-MSE for room type, room addition, room removal,
and opening deletion.

An editing-aware Raster2Seq wrapper with the same outside-freeze constraint
reaches only 0.1634 Delta-IoU and 0.0732 Delta-MSE, so FloorFlow's advantage is
not solely explained by the locality wrapper.

## Explicit boundaries

- This is not a strict HouseMind RPLAN leaderboard result.
- The dual-metric and structural results are CubiCasa protocol evidence, not a
  HouseMind RPLAN leaderboard result.
- Raster2Seq is a reconstruction baseline without edit awareness.
- Furniture deletion is supplementary because Raster2Seq has no furniture output.
- Exact match means strict `uint8` label equality with no tolerance, after
  excluding the edit rectangle, footprint exterior, and dilated walls.
- Missing-control robustness, predicted controls, distribution coverage, and
  full-CAD external superiority are not claimed.
- The control gate is the audited locality mechanism; the watertight guard has
  no stable aggregate gain.
- The typed graph ablation is post-hoc routing, not retrained cross-attention.
- Wall-move, room-resize, and multi-step edits are explicit coverage gaps.
- The matched Pareto excludes HouseMind-E; its anchor is cross-dataset context.

## Structural appendix

The canonical graph protocol applies nearest-room footprint tiling and
connected-polygon extraction symmetrically to prediction and target. Across
five seeds, Node F1 is 0.9281-0.9376, edge overlap is 0.8777-0.8978, and unit
GED is 1.696-1.969. Target-matched wall-junction F1 is 0.9397-0.9611 and
junction-degree exactness is 0.9377-0.9716.

The HouseMind-style after-edit view reaches Node F1 0.9802-0.9828, edge overlap
0.9228-0.9325, and GED 0.8245-0.9835. It is protocol-aligned oracle-control
evidence, not an RPLAN leaderboard rank. The canonical instance graph remains a
secondary diagnostic. Raw wall continuity is not a headline because a sparser
wall output can inflate it.

## Strict-locality evidence

- Wall-adjacent, opening-covering, and exterior-boundary stress suites each
  contain 2,055-2,214 cases per seed.
- Raw outside exactness is 0.9661-0.9910; the control gate restores exact
  pixel and room-label preservation to 1.0.
- Validity remains 1.0 and inside-edit response remains about 0.32 of edit-valid
  pixels, so locality is not achieved by suppressing editing.
- Room-count, edit-area, and interface strata combine 12,395 rows across five
  seeds. Strict locality and validity hold in every supported bin.
- Conditional watertightness is not uniform: the 0-4-room, opening-only, and
  neither-interface bins remain below 0.95 and are disclosed.

The HouseMind-E comparison is protocol-coverage only. HouseMind-E uses RPLAN
natural-language editing; FloorFlow uses CubiCasa control-gated editing. No
same-benchmark rank is claimed.

## External comparison provenance

- Method: Raster2Seq, code commit
  `a6c4e27a68d11d7a459f6e4a2601fd887227dd1a`
- Checkpoint:
  `https://huggingface.co/haopt/Raster2Seq`
- Checkpoint SHA-256:
  `f7c0ef9379aa8cf11349da8b2179214e0d495af758383984a2361a1309aab3b7`
- Pairs: 2,479
- Independent inferences: 4,958
- Per-case rows:
  `artifacts/outputs/external_baselines/raster2seq/test503_matched_v1/report.json`

## Rerun

```bash
uv sync --extra dev --extra train
uv pip install -r configs/fp_raster2seq_requirements.txt
uv run python scripts/audit_fp_housemind_reproduction.py
uv run python scripts/build_fp_multiseed_report.py \
  --run 19=outputs/eval/fp_hard_metrics_full_test503_seed19_v1/metrics.json:outputs/eval/fp_hard_metrics_full_test503_seed19_v1/per_case.json \
  --run 20=outputs/eval/fp_hard_metrics_full_test503_seed20_v1/metrics.json:outputs/eval/fp_hard_metrics_full_test503_seed20_v1/per_case.json \
  --run 21=outputs/eval/fp_hard_metrics_full_test503_seed21_v1/metrics.json:outputs/eval/fp_hard_metrics_full_test503_seed21_v1/per_case.json \
  --run 22=outputs/eval/fp_hard_metrics_full_test503_seed22_v1/metrics.json:outputs/eval/fp_hard_metrics_full_test503_seed22_v1/per_case.json \
  --run 23=outputs/eval/fp_hard_metrics_full_test503_seed23_v1/metrics.json:outputs/eval/fp_hard_metrics_full_test503_seed23_v1/per_case.json \
  --output outputs/reports/fp_hard_metrics_multiseed_test503_v2.json
uv run python scripts/build_fp_edit_detection_multiseed_report.py \
  --seed-report outputs/reports/fp_edit_detection_test503_seed19_v2.json \
  --seed-report outputs/reports/fp_edit_detection_test503_seed20_v2.json \
  --seed-report outputs/reports/fp_edit_detection_test503_seed21_v2.json \
  --seed-report outputs/reports/fp_edit_detection_test503_seed22_v2.json \
  --seed-report outputs/reports/fp_edit_detection_test503_seed23_v2.json \
  --output outputs/reports/fp_edit_detection_test503_fiveseed_v2.json
uv run python scripts/build_fp_raster2seq_matched_report.py
for seed in 19 20 21 22 23; do
  uv run python scripts/build_fp_delta_mse_decomposition_report.py \
    --seed "${seed}" \
    --output "outputs/reports/fp_delta_mse_decomposition_test503_seed${seed}_v1.json"
done
uv run python scripts/build_fp_delta_mse_multiseed_report.py \
  --seed-report outputs/reports/fp_delta_mse_decomposition_test503_seed19_v1.json \
  --seed-report outputs/reports/fp_delta_mse_decomposition_test503_seed20_v1.json \
  --seed-report outputs/reports/fp_delta_mse_decomposition_test503_seed21_v1.json \
  --seed-report outputs/reports/fp_delta_mse_decomposition_test503_seed22_v1.json \
  --seed-report outputs/reports/fp_delta_mse_decomposition_test503_seed23_v1.json \
  --output outputs/reports/fp_delta_mse_decomposition_test503_fiveseed_v1.json \
  --table paper/tables/fp_delta_mse_decomposition.md
uv run python scripts/build_fp_delta_mse_gate_report.py
uv run python scripts/audit_fp_room_graph_sources.py --seed 19
uv run python scripts/build_fp_room_graph_protocol_report.py \
  --readout direct_tiled --prediction-key prediction
uv run python scripts/build_fp_canonical_structural_multiseed_report.py \
  --graph-report outputs/reports/fp_room_graph_matching_protocol_final_v1.json \
  --graph-report outputs/reports/fp_room_graph_matching_protocol_final_seed20_v1.json \
  --graph-report outputs/reports/fp_room_graph_matching_protocol_final_seed21_v1.json \
  --graph-report outputs/reports/fp_room_graph_matching_protocol_final_seed22_v1.json \
  --graph-report outputs/reports/fp_room_graph_matching_protocol_final_seed23_v1.json \
  --source-report outputs/reports/fp_room_graph_source_audit_v1.json \
  --source-report outputs/reports/fp_room_graph_source_audit_seed20_v1.json \
  --source-report outputs/reports/fp_room_graph_source_audit_seed21_v1.json \
  --source-report outputs/reports/fp_room_graph_source_audit_seed22_v1.json \
  --source-report outputs/reports/fp_room_graph_source_audit_seed23_v1.json \
  --output outputs/reports/fp_canonical_structural_fiveseed_v1.json \
  --table paper/tables/fp_canonical_structural_metrics.md
uv run python scripts/audit_fp_wall_junction_continuity.py
uv run python scripts/build_fp_housemind_structural_alignment.py
uv run python scripts/build_fp_complexity_strata.py
uv run python scripts/build_fp_strict_locality_stress.py
uv run python scripts/build_fp_failure_analysis_v2.py
uv run python scripts/build_fp_architecture_necessity_ablation.py
uv run python scripts/build_fp_cvpr_evidence.py --output outputs/reports/fp_cvpr_evidence_v4.json
uv run python scripts/build_fp_evidence_v5.py
uv run python scripts/build_fp_housemind_public_status_v2.py
uv run python scripts/build_fp_protocol_v2.py
uv run python scripts/build_fp_typed_edit_graph_ablation.py \
  --seed 19 --seed 20 --seed 21 --seed 22 --seed 23 --workers 24
uv run python scripts/build_fp_edit_type_strata_v2.py
uv run python scripts/build_fp_dual_pareto_v2.py
uv run python scripts/build_fp_evidence_v6.py
uv run python scripts/build_reviewer_risk_matrix_v3.py
uv run pytest -q --junitxml=outputs/reports/fp_pytest_v11.xml
```

Raster2Seq inference additionally requires its official CUDA extensions. Build
them from `third_party/Raster2Seq/models/ops` and
`third_party/Raster2Seq/diff_ras` with `CUDA_HOME` set. Then run:

```bash
uv run python scripts/run_fp_raster2seq_matched_edits.py \
  --batch-size 16 --device cuda:0 --skip-existing \
  --output-dir outputs/external_baselines/raster2seq/test503_matched_v1
```

The checkpoint and CubiCasa data are not embedded in this archive because they
are external assets. Their hashes and official URLs are recorded in the JSON
reports.

The `data/` directory contains all 12,395 rendered control-gated output files,
Raster2Seq token-rendered predictions, and available seed training configs.
Control-change gating operates on rendered pre/post predictions and introduces
no new network logits; this is recorded explicitly in the manifest.
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")
    archive_path = Path("outputs/cvpr_package_fp_v11.tar.gz")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(package_dir, arcname=package_dir.name)
    Path(f"{archive_path}.sha256").write_text(
        f"{digest(archive_path)}  {archive_path}\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Build a deterministic residential-only CVPR experiment archive."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "residential_sota_evidence_v4"
PACKAGE_ROOT = ROOT / "outputs" / PACKAGE_NAME
ARCHIVE = ROOT / "outputs" / f"{PACKAGE_NAME}.tar.gz"

FILES = (
    "fptodo.json",
    "pyproject.toml",
    "uv.lock",
    "configs/residential_sota_lockbox_v1.json",
    "configs/residential_identity_selected_v1.json",
    "configs/residential_identity_consensus_v1.json",
    "configs/residential_identity_dihedral_consensus_v1.json",
    "configs/residential_room_iou_fusion_v1.json",
    "configs/residential_identity_finetune_v1.json",
    "configs/fevp_metric_contract_v2.json",
    "configs/fevp2_vector_contract_v1.json",
    "src/floorflow/inference/local_delta.py",
    "src/floorflow/inference/raster2seq_identity_refiner.py",
    "src/floorflow/inference/housemind_free_text_editor.py",
    "src/floorflow/evaluation/paired_bootstrap.py",
    "src/floorflow/evaluation/statistical_effect_sizes.py",
    "src/floorflow/evaluation/raster2seq_polygon_fusion.py",
    "src/floorflow/evaluation/housemind_free_text_metrics.py",
    "scripts/build_raster2seq_tta_dataset_v1.py",
    "scripts/evaluate_raster2seq_dihedral_consensus_v1.py",
    "scripts/evaluate_raster2seq_tta_consensus_v1.py",
    "scripts/evaluate_raster2seq_identity_refiner_v1.py",
    "scripts/freeze_raster2seq_polygon_fusion_v1.py",
    "scripts/evaluate_raster2seq_polygon_fusion_v1.py",
    "scripts/evaluate_housemind_free_text_editor_v1.py",
    "scripts/probe_housemind_weights_v1.py",
    "scripts/build_housemind_inference_feasibility_v1.py",
    "scripts/build_residential_graph_protocol_dominance_v1.py",
    "scripts/build_rplan_full_cad_sota_gate_v1.py",
    "scripts/build_residential_aggressive_sota_gate_v1.py",
    "scripts/build_raster2seq_sensitivity_v1.py",
    "scripts/build_ce2eplan_tvcg_anchor_v2.py",
    "scripts/run_housemind_partial_reproduction_v1.py",
    "scripts/build_housemind_partial_table_v1.py",
    "scripts/build_residential_blocker_rereview_v2.py",
    "scripts/evaluate_control_edit_baseline_v1.py",
    "scripts/evaluate_floorflow_raster2seq_hybrid_v1.py",
    "scripts/evaluate_fp_raster2seq_local_redraw.py",
    "scripts/build_residential_identity_report_v3.py",
    "scripts/build_residential_identity_report_v4.py",
    "scripts/build_residential_identity_report_v1.py",
    "scripts/build_residential_edit_fiveseed_report_v1.py",
    "scripts/build_residential_edit_vector_protocol_v1.py",
    "scripts/build_residential_structural_hardened_v2.py",
    "scripts/build_residential_sota_main_table_v2.py",
    "scripts/build_residential_sota_ablation_v1.py",
    "scripts/build_residential_sota_strata_v1.py",
    "scripts/build_residential_sota_protocol_attacks_v1.py",
    "scripts/build_residential_sota_controls_v1.py",
    "scripts/build_residential_sota_claim_gate_v1.py",
    "scripts/build_residential_sota_provenance_v1.py",
    "scripts/build_residential_sota_evidence_v4.py",
    "scripts/audit_residential_sota_evidence_v4.py",
    "scripts/audit_python_line_limit_v1.py",
    "scripts/blend_raster2seq_checkpoints_v1.py",
    "scripts/blend_raster2seq_weighted_v1.py",
    "tests/test_local_delta.py",
    "tests/test_raster2seq_identity_refiner.py",
    "tests/test_raster2seq_tta_dataset.py",
    "tests/test_raster2seq_polygon_fusion.py",
    "tests/test_housemind_free_text_editor.py",
    "tests/test_raster2seq_checkpoint_blend.py",
    "third_party/Raster2Seq/data_preprocess/cubicasa5k/create_coco_cc5k.py",
    "third_party/Raster2Seq/data_preprocess/cubicasa5k/house.py",
    "third_party/Raster2Seq/data_preprocess/cubicasa5k/floorplan_extraction.py",
    "third_party/Raster2Seq/data_preprocess/cubicasa5k/combine_json.py",
    "third_party/Raster2Seq/eval.py",
    "outputs/reports/residential_identity_selection_val_v1.json",
    "outputs/reports/residential_identity_test509_v1.json",
    "outputs/reports/residential_identity_test509_v2.json",
    "outputs/reports/residential_identity_test509_v2_summary.json",
    "outputs/reports/raster2seq_tta_consensus_val_v1.json",
    "outputs/reports/raster2seq_dihedral_consensus_val_v1.json",
    "outputs/reports/residential_identity_test509_v3_dihedral.json",
    "outputs/reports/residential_identity_test509_v3_summary.json",
    "outputs/reports/raster2seq_polygon_fusion_val_v3.json",
    "outputs/reports/raster2seq_polygon_fusion_val_v4.json",
    "outputs/reports/raster2seq_polygon_fusion_val_v5.json",
    "outputs/reports/residential_identity_test509_room_iou_v1.json",
    "outputs/reports/residential_identity_test509_v4_summary.json",
    "outputs/reports/residential_edit_test503_v1.json",
    "outputs/reports/residential_edit_vector_protocol_beforeonly_v1.json",
    "outputs/reports/residential_edit_vector_protocol_raw_v1.json",
    "outputs/reports/residential_editing_aware_baseline_v1.json",
    "outputs/reports/residential_structural_hardened_v2.json",
    "outputs/reports/residential_validity_reconciliation_v2.json",
    "outputs/reports/residential_sota_main_results_v4.json",
    "outputs/reports/residential_sota_ablation_v1.json",
    "outputs/reports/residential_sota_strata_v1.json",
    "outputs/reports/residential_sota_failures_v1.json",
    "outputs/reports/residential_sota_protocol_attacks_v1.json",
    "outputs/reports/residential_sota_controls_v1.json",
    "outputs/reports/residential_sota_claim_gate_v5.json",
    "outputs/reports/residential_sota_provenance_v4.json",
    "outputs/reports/residential_line_limit_audit_v4.json",
    "outputs/reports/fp_control_gated_hard_gate_v1.json",
    "outputs/reports/fp_strict_locality_stress_fiveseed_v1.json",
    "outputs/reports/fevp_fiveseed_stability_v2.json",
    "outputs/reports/fevp_graph_metrics_v2.json",
    "outputs/reports/fevp_validity_watertight_v2.json",
    "outputs/reports/fevp_protocol_attacks_v2.json",
    "outputs/reports/fp_canonical_structural_fiveseed_v1.json",
    "outputs/reports/fp_housemind_structural_alignment_v1.json",
    "outputs/reports/fp_wall_junction_continuity_fiveseed_v1.json",
    "outputs/reports/fevp_protocol_attacks_v2.json",
    "outputs/reports/housemind_released_asset_audit_v1.json",
    "outputs/reports/housemind_weight_probe_v1.json",
    "outputs/reports/housemind_inference_feasibility_v1.json",
    "outputs/reports/housemind_partial_reproduction_v1.json",
    "outputs/reports/housemind_partial_table_v1.json",
    "outputs/reports/housemind_free_text_editor_holdout_v1.json",
    "outputs/reports/residential_graph_protocol_dominance_v1.json",
    "outputs/reports/rplan_full_cad_sota_gate_v1.json",
    "outputs/reports/residential_aggressive_sota_gate_v1.json",
    "outputs/reports/raster2seq_sensitivity_v1.json",
    "outputs/reports/ce2eplan_tvcg_anchor_v2.json",
    "outputs/reports/residential_sota_main_results_v5.json",
    "outputs/reports/residential_blocker_rereview_v2.json",
    "outputs/reports/raster2seq_ensemble_val_v1.json",
    "outputs/reports/raster2seq_blend_selection_val_v1.json",
    "outputs/reports/raster2seq_identity_refiner_val_v1.json",
    "paper/tables/residential_identity_main_v1.md",
    "paper/tables/residential_identity_main_v2.md",
    "paper/tables/residential_identity_main_v3.md",
    "paper/tables/residential_identity_main_v4.md",
    "paper/tables/residential_edit_fiveseed_v1.md",
    "paper/tables/residential_edit_ablation_v1.md",
    "paper/tables/residential_sota_main_v1.md",
    "paper/tables/residential_sota_main_v2.md",
    "paper/tables/residential_sota_main_v3.md",
    "paper/tables/residential_sota_main_v4.md",
    "paper/tables/residential_sota_main_v5.md",
    "paper/tables/residential_structural_v2.md",
    "paper/tables/residential_graph_protocol_dominance_v1.md",
    "paper/tables/residential_aggressive_sota_gate_v1.md",
    "paper/tables/housemind_partial_reproduction_v1.md",
    "paper/tables/raster2seq_sensitivity_v1.md",
    "paper/tables/ce2eplan_tvcg_anchor_v2.md",
    "paper/tables/residential_blocker_rereview_v2.md",
    "docs/housemind_matched_comparison_boundary_v1.md",
    "docs/rplan_full_cad_claim_boundary_v1.md",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def collect_paths() -> list[Path]:
    paths = [ROOT / name for name in FILES]
    runs = (
        ROOT
        / "outputs/external_baselines/raster2seq/official_eval_val503_blend_0.75_v1",
        ROOT
        / "outputs/external_baselines/raster2seq/residential_identity_test509_alpha075_v1",
    )
    transforms = (
        "hflip",
        "vflip",
        "rot180",
        "rot90",
        "rot270",
        "transpose",
        "transverse",
    )
    for split in ("val503", "test509"):
        for transform in transforms:
            runs += (
                ROOT
                / (
                    "outputs/external_baselines/raster2seq/"
                    f"tta_{split}_{transform}_alpha075_v1"
                ),
            )
    for run in runs:
        paths.append(run / "results.txt")
        paths.extend(sorted(run.glob("jsons/*.json")))
    paths.extend(
        sorted(
            (ROOT / "outputs/reports").glob(
                "residential_edit_hybrid_typedall_seed*_v1.json"
            )
        )
    )
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")
    if any("chip" in str(path).lower() for path in paths):
        raise ValueError("residential package must not contain chips artifacts")
    return sorted(set(paths))


def add_file(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(path), arcname=arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    with path.open("rb") as handle:
        tar.addfile(info, handle)


def main() -> None:
    paths = collect_paths()
    manifest_files = []
    with tempfile.TemporaryDirectory(prefix="residential-evidence-") as temporary:
        stage = Path(temporary)
        for path in paths:
            relative = path.relative_to(ROOT)
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            manifest_files.append(
                {
                    "path": f"{PACKAGE_NAME}/{relative}",
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        provenance = json.loads(
            (
                ROOT / "outputs/reports/residential_sota_provenance_v4.json"
            ).read_text(encoding="utf-8")
        )
        manifest = {
            "package": PACKAGE_NAME,
            "status": "complete",
            "scope": "residential_floorplan_only",
            "chips_artifacts_included": False,
            "manifest_excludes_itself": True,
            "files": manifest_files,
            "external_artifacts": {
                "public_raster2seq_source_checkpoint": {
                    "path": "outputs/external_baselines/raster2seq/cubicasa5k/checkpoint.pth",
                    "sha256": "f7c0ef9379aa8cf11349da8b2179214e0d495af758383984a2361a1309aab3b7",
                },
                "selected_identity_checkpoint": {
                    "path": "outputs/external_baselines/raster2seq/checkpoint_blends_v2/ema_alpha_0.75.pth",
                    "sha256": "5c738ba5cda8842856a945aba67e60e8b373a8f5c86047af0c1c23c19e362311",
                },
                "official_test509_tree_sha256": (
                    "d333f2873e70807a0f78b4043d0f0cfd13fa6c7fb8e74b74972de759d657fd4f"
                ),
                "house_mind_benchmark": {
                    "path": "/tmp/housemind_data_test.zip",
                    "sha256": (
                        "a0aac3189063c72a00977bbb7479fd3d01eacd02c298108305025d6fc557e136"
                    ),
                    "redistributed": False,
                },
                "house_mind_weights": {
                    "path": "/tmp/housemind_weights.zip",
                    "sha256": (
                        "1fe7235dbe0945a3d612a1ccc0828169fd0310aa3ff4fb693b279266e45ba119"
                    ),
                    "redistributed": False,
                },
            },
            "environment": provenance["environment"],
            "raster2seq_commit": provenance["raster2seq"]["commit"],
        }
        checksums = "\n".join(
            f"{row['sha256']}  {row['path']}" for row in manifest_files
        )
        checksum_path = stage / "SHA256SUMS"
        checksum_path.write_text(checksums + "\n", encoding="utf-8")
        manifest_files.append(
            {
                "path": f"{PACKAGE_NAME}/SHA256SUMS",
                "sha256": sha256(checksum_path),
                "bytes": checksum_path.stat().st_size,
            }
        )
        manifest["files"] = manifest_files
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        if PACKAGE_ROOT.exists():
            shutil.rmtree(PACKAGE_ROOT)
        shutil.copytree(stage, PACKAGE_ROOT)
        archive_files = sorted(
            path for path in PACKAGE_ROOT.rglob("*") if path.is_file()
        )
        archive_files.append(PACKAGE_ROOT / "manifest.json")
        archive_files = sorted(set(archive_files))
        with (
            ARCHIVE.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz,
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
        ):
            for path in archive_files:
                arcname = f"{PACKAGE_NAME}/{path.relative_to(PACKAGE_ROOT)}"
                add_file(tar, path, arcname)
    print(
        json.dumps(
            {
                "status": "complete",
                "files": len(manifest_files),
                "package": str(PACKAGE_ROOT.relative_to(ROOT)),
                "archive": str(ARCHIVE.relative_to(ROOT)),
                "archive_sha256": sha256(ARCHIVE),
                "archive_bytes": ARCHIVE.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

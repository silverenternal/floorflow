"""Assemble the deterministic residential CAGN evidence package."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from floorflow.evaluation.deterministic_archive import (
    build_deterministic_tar_gz,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/residential_cagn_evidence_v1"
ARCHIVE = ROOT / "outputs/residential_cagn_evidence_v1.tar.gz"
REQUIRED = (
    "outputs/reports/residential_constraint_action_audit_v1.json",
    "outputs/reports/residential_constraint_embedding_v1.json",
    "outputs/reports/residential_cagn_training_screen_v1.json",
    "outputs/reports/residential_cagn_validation_v3_seed19.json",
    "outputs/reports/residential_cagn_multiseed_v1.json",
    "outputs/reports/residential_differentiable_fevp_audit_v1.json",
    "outputs/reports/residential_soft_ged_audit_v1.json",
    "outputs/reports/residential_cagn_five_arm_v1.json",
    "outputs/reports/residential_cagn_stress_v1.json",
    "outputs/reports/residential_cagn_validation_freeze_v1.json",
    "outputs/reports/residential_cagn_test503_final_v1.json",
    "paper/tables/residential_cagn_ablation_v1.md",
    "paper/tables/residential_cagn_stress_v1.md",
    "configs/residential_constraint_action_v1.json",
    "configs/residential_constraint_embedding_v1.json",
    "configs/residential_cagn_joint_v1.json",
    "schema/residential-constraint-action-v1.schema.json",
    "src/floorflow/data/constraint_action_dataset.py",
    "src/floorflow/data/constraint_candidate_dataset.py",
    "src/floorflow/data/constraint_candidates.py",
    "src/floorflow/model/constraint_state_encoder.py",
    "src/floorflow/model/constraint_aware_gate.py",
    "src/floorflow/objectives/constraint_contrastive.py",
    "src/floorflow/objectives/constraint_gate_loss.py",
    "src/floorflow/objectives/differentiable_fevp.py",
    "src/floorflow/objectives/soft_graph_edit.py",
    "src/floorflow/evaluation/constraint_calibration.py",
    "src/floorflow/evaluation/constraint_gate_runtime.py",
    "scripts/train_constraint_embedding.py",
    "scripts/train_constraint_aware_gate.py",
    "scripts/freeze_residential_cagn.py",
    "scripts/evaluate_residential_cagn_test503.py",
    "outputs/control_contracts/oracle_cagn_test_v1/test/manifest.json",
    "outputs/control_contracts/residential_cagn_v1/test/index.json",
    "outputs/control_contracts/residential_cagn_features_v1/test/features.pt",
)


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _copy(relative: str) -> dict:
    source = ROOT / relative
    if not source.exists():
        raise FileNotFoundError(f"required evidence missing: {relative}")
    target = PACKAGE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "path": relative,
        "source": relative,
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _checkpoint_rows() -> list[dict]:
    rows = []
    for pattern in (
        "outputs/residential_cagn_embedding_v1/seed19.pt",
        "outputs/residential_cagn_gate_v3/seed*.pt",
        "outputs/residential_cagn_gate_scratch_v3/seed*.pt",
    ):
        for path in sorted(ROOT.glob(pattern)):
            relative = str(path.relative_to(ROOT))
            target = PACKAGE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            rows.append(
                {
                    "path": relative,
                    "source": relative,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return rows


def main() -> None:
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    files = [_copy(path) for path in REQUIRED]
    files.extend(_checkpoint_rows())
    reports = {
        "audit": _read(REQUIRED[0]),
        "embedding": _read(REQUIRED[1]),
        "training_screen": _read(REQUIRED[2]),
        "gate": _read(REQUIRED[3]),
        "multiseed": _read(REQUIRED[4]),
        "differentiable_fevp": _read(REQUIRED[5]),
        "soft_ged": _read(REQUIRED[6]),
        "ablation": _read(REQUIRED[7]),
        "stress": _read(REQUIRED[8]),
        "freeze": _read(REQUIRED[9]),
        "test503": _read(REQUIRED[10]),
    }
    failed = {
        name: report.get("status")
        for name, report in reports.items()
        if report.get("status") not in {"passed", "frozen"}
    }
    claim_gate = {
        "schema_version": "residential-cagn-claim-gate-v1",
        "status": "passed" if not failed else "failed",
        "scope": "residential_floorplan_only",
        "no_chips_access": True,
        "test503_evaluated_once": True,
        "soft_metrics_are_training_proxies_only": True,
        "component_status": {
            name: report.get("status")
            for name, report in reports.items()
        },
        "failed_components": failed,
    }
    claim_path = PACKAGE / "outputs/reports/residential_cagn_claim_gate_v1.json"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text(
        json.dumps(claim_gate, indent=2) + "\n",
        encoding="utf-8",
    )
    root_claim = ROOT / "outputs/reports/residential_cagn_claim_gate_v1.json"
    root_claim.write_text(
        json.dumps(claim_gate, indent=2) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "residential-cagn-provenance-v1",
        "timestamp_policy": "omitted to preserve byte-deterministic archives",
        "scope": "residential_floorplan_only",
        "gpu": "cuda:0",
        "test_policy": "single evaluation after validation freeze",
        "source_inputs": files,
        "source_digest": hashlib.sha256(
            "".join(sorted(row["sha256"] for row in files)).encode("ascii")
        ).hexdigest(),
    }
    provenance_path = PACKAGE / "outputs/reports/residential_cagn_provenance_v1.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    root_provenance = (
        ROOT / "outputs/reports/residential_cagn_provenance_v1.json"
    )
    root_provenance.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    files.extend(
        [
            {
                "path": str(claim_path.relative_to(PACKAGE)),
                "source": str(claim_path.relative_to(ROOT)),
                "sha256": sha256_file(claim_path),
                "bytes": claim_path.stat().st_size,
            },
            {
                "path": str(provenance_path.relative_to(PACKAGE)),
                "source": str(provenance_path.relative_to(ROOT)),
                "sha256": sha256_file(provenance_path),
                "bytes": provenance_path.stat().st_size,
            },
        ]
    )
    manifest = {
        "schema_version": "residential-cagn-package-v1",
        "files": files,
        "limits": {
            "residential_only": True,
            "no_chips": True,
            "soft_metrics_not_fevp": True,
        },
    }
    manifest_path = PACKAGE / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_row = {
        "path": "manifest.json",
        "source": "generated",
        "sha256": sha256_file(manifest_path),
        "bytes": manifest_path.stat().st_size,
    }
    archive_hash = build_deterministic_tar_gz(PACKAGE, ARCHIVE)
    with tempfile.TemporaryDirectory() as directory:
        rebuilt = Path(directory) / ARCHIVE.name
        rebuilt_hash = build_deterministic_tar_gz(PACKAGE, rebuilt)
    with tarfile.open(ARCHIVE, "r:gz") as archive_handle:
        archive_paths = sorted(archive_handle.getnames())
    expected_paths = sorted(
        [f"{PACKAGE.name}/{row['path']}" for row in files]
        + [f"{PACKAGE.name}/manifest.json"]
    )
    contents_match = archive_paths == expected_paths
    audit = {
        "schema_version": "residential-cagn-package-audit-v1",
        "status": (
            "passed"
            if (
                archive_hash == rebuilt_hash
                and contents_match
                and claim_gate["status"] == "passed"
            )
            else "failed"
        ),
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": archive_hash,
        "rebuilt_sha256": rebuilt_hash,
        "manifest": manifest_row,
        "files": len(files),
        "archive_matches_manifest": contents_match,
    }
    (ROOT / "outputs/reports/residential_cagn_package_audit_v1.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

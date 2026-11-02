#!/usr/bin/env python3
"""Freeze either the scoped existing evidence or an optional fresh held-out run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPS = ROOT / "configs/predicted_operation_success_contract_v1.json"
DEFAULT_REGISTRY = ROOT / "docs/floorplan_submission_source_registry_v2.json"
EVALUATOR_FILES = (
    "src/floorflow/evaluation/fevp_metric_contracts.py",
    "src/floorflow/evaluation/predicted_edit_graph_metrics.py",
    "src/floorflow/evaluation/structural_legality.py",
    "src/floorflow/evaluation/predicted_operation_success.py",
    "configs/fevp_metric_contract_v2.json",
    "configs/housemind_style_ged_deterministic_v2.json",
    "configs/predicted_control_reasonableness_contract_v1.json",
    "configs/residential_predicted_edit_v1.json",
)
RELEASE_SOURCE_FILES = (
    "scripts/build_floorplan_submission_evidence_v2.py",
    "scripts/build_floorplan_submission_tables_v2.py",
    "scripts/build_floorplan_submission_figures_v1.py",
    "scripts/audit_floorplan_evidence_freeze_v2.py",
    "scripts/audit_floorplan_submission_integrity_v1.py",
)
SCOPED_CHECKPOINTS = (
    "outputs/train/raster_room_control_native_full_v1/best.pt",
    "outputs/train/raster_structure_bridge_full_v1/best.pt",
    "outputs/train/raster_structure_bridge_full_seed20_v1/best.pt",
    "outputs/train/raster_structure_bridge_full_seed21_v1/best.pt",
    "outputs/train/raster_structure_bridge_full_seed22_v1/best.pt",
    "outputs/train/raster_structure_bridge_full_seed23_v1/best.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("existing_evidence_scoped", "optional_fresh_heldout"),
        default="existing_evidence_scoped",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--operation-contract", type=Path, default=DEFAULT_OPS)
    parser.add_argument("--heldout-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return commit, dirty


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    return {
        "path": str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved),
        "exists": resolved.is_file(),
        "bytes": resolved.stat().st_size if resolved.is_file() else None,
        "sha256": _sha256(resolved) if resolved.is_file() else None,
        "mtime_utc": datetime.fromtimestamp(resolved.stat().st_mtime, UTC).isoformat()
        if resolved.is_file()
        else None,
    }


def _required_files(paths: tuple[Path | None, ...], mode: str) -> list[str]:
    missing = [str(path) for path in paths if path is None or not path.is_file()]
    if mode == "optional_fresh_heldout":
        required = (
            "checkpoint",
            "config",
            "prompt manifest",
            "taxonomy",
            "operation contract",
            "held-out manifest",
        )
        if missing:
            return [f"missing {name}: {path or '<not supplied>'}" for name, path in zip(required, paths, strict=True)]
    return [f"missing required input: {path}" for path in missing if path is not None]


def _existing_evidence_freeze(args: argparse.Namespace, commit: str, dirty: bool) -> dict[str, Any]:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry.get("status") != "passed_scoped_release":
        raise SystemExit("source registry must have status=passed_scoped_release")
    source_files = {
        key: _file_identity(ROOT / row["path"])
        for key, row in registry["source_files"].items()
    }
    missing_sources = [key for key, row in source_files.items() if not row["exists"]]
    if missing_sources:
        raise SystemExit(f"missing registry source files: {missing_sources}")
    checkpoints = {name: _file_identity(Path(name)) for name in SCOPED_CHECKPOINTS}
    missing_checkpoints = [name for name, row in checkpoints.items() if not row["exists"]]
    evaluators = {name: _file_identity(Path(name)) for name in EVALUATOR_FILES}
    release_sources = {
        name: _file_identity(Path(name))
        for name in RELEASE_SOURCE_FILES
        if (ROOT / name).is_file()
    }
    threshold_contract = json.loads((ROOT / "configs/predicted_control_reasonableness_contract_v1.json").read_text())
    return {
        "schema_version": "floorplan-submission-evidence-freeze-v2",
        "mode": "existing_evidence_scoped",
        "status": "frozen_scoped_release",
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit_at_freeze": commit,
        "worktree_dirty_at_freeze": dirty,
        "clean_worktree_required": False,
        "clean_worktree_policy": (
            "The repository contains unrelated active research changes. Freeze the scoped evidence and "
            "release files by identity rather than misrepresenting the whole worktree as clean."
        ),
        "scope": {
            "dataset": "CubiCasa5K",
            "oracle_split": "test503 development evaluation",
            "predicted_split": "val491 five-seed target-free validation",
            "operation_split": "test503 development-exposed prompts, not held-out",
            "semantic_split": "val191 diagnostic rows from an older model",
            "generalization": "optional_fresh_heldout_not_evaluated",
        },
        "registry": _file_identity(args.registry),
        "source_files": source_files,
        "checkpoints": checkpoints,
        "evaluators": evaluators,
        "release_sources": release_sources,
        "thresholds": {
            "reasonableness_contract": threshold_contract.get("thresholds"),
            "operation_contract": _file_identity(DEFAULT_OPS),
            "semantic_0_90": "historical exploratory target only; not a release prerequisite",
            "scoped_release_uses_observed_counts": True,
        },
        "statistical_policy": {
            "oracle_unit": "floorplan cluster; repeated seeds retained separately",
            "predicted_unit": "491 validation cases inside each repeated execution seed",
            "operation_unit": "prompt; safety and semantic realization reported separately",
            "pooling": "oracle, predicted, identity, and external rows are never pooled",
        },
        "historical_execution_hashes": {
            "available": False,
            "limitation": (
                "The saved reports do not embed a verifiable single execution manifest for every seed. "
                "Checkpoint, config, report, evaluator, and current release-source hashes are frozen here; "
                "they are not presented as historical execution hashes."
            ),
        },
        "heldout": {
            "status": "optional_generalization_not_evaluated",
            "blocks_scoped_submission": False,
            "requires_pre_access_freeze_if_used": True,
        },
        "freeze_policy": [
            "Do not replace scoped rows with newer developmental model outputs.",
            "Do not convert abstentions, no-ops, or contract checks into semantic success.",
            "Do not describe the development-exposed test prompts as held-out.",
            "Preserve raw and functional watertightness as separate metrics.",
            "Freeze prompts, thresholds, and evaluator identity before any optional held-out access.",
        ],
        "missing_optional_inputs": missing_checkpoints,
    }


def main() -> None:
    args = parse_args()
    commit, dirty = _git_state()
    if args.mode == "optional_fresh_heldout":
        required = (
            args.checkpoint,
            args.config,
            args.prompt_manifest,
            args.taxonomy,
            args.operation_contract,
            args.heldout_manifest,
        )
        missing = _required_files(required, args.mode)
        if missing:
            raise SystemExit(f"missing freeze inputs: {missing}")
        if dirty and not args.allow_dirty:
            raise SystemExit(
                "refusing optional held-out freeze from a dirty worktree; commit the selected model and evaluator "
                "or explicitly pass --allow-dirty for a non-authoritative draft"
            )
        payload: dict[str, Any] = {
            "schema_version": "predicted-control-submission-freeze-v1",
            "mode": "optional_fresh_heldout",
            "status": "frozen_before_heldout_access",
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": commit,
            "worktree_dirty": dirty,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "config": str(args.config),
            "config_sha256": _sha256(args.config),
            "prompt_manifest": str(args.prompt_manifest),
            "prompt_manifest_sha256": _sha256(args.prompt_manifest),
            "taxonomy_sha256": _sha256(args.taxonomy),
            "operation_success_contract": str(args.operation_contract),
            "operation_success_contract_sha256": _sha256(args.operation_contract),
            "evaluator_sha256": {name: _sha256(ROOT / name) for name in EVALUATOR_FILES},
            "heldout_manifest": str(args.heldout_manifest),
            "heldout_manifest_sha256": _sha256(args.heldout_manifest),
            "thresholds": {
                "operation_semantic_success": 0.9,
                "unsupported_abstention": 1.0,
                "outside_pixel_exact": 1.0,
                "outside_room_label_exact": 1.0,
                "validity": 1.0,
                "plan_plausibility": 0.95,
                "functional_watertightness": 0.95,
            },
            "tolerances": {
                "room_match_min_iou": 0.5,
                "room_polygon_hausdorff_px": 0.5,
                "edge_contact_px": 2.0,
                "opening_center_px": 1.0,
                "changed_pixel_epsilon": 0.0,
            },
            "freeze_policy": [
                "No checkpoint or threshold changes after held-out access.",
                "Held-out is executed once.",
                "Negative and failed metrics remain in the report.",
            ],
        }
    else:
        payload = _existing_evidence_freeze(args, commit, dirty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mode": args.mode, "status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()

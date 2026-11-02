#!/usr/bin/env python3
"""Prepare an immutable predicted-control submission freeze manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPS = ROOT / "configs/predicted_operation_success_contract_v1.json"
EVALUATOR_FILES = (
    "src/floorflow/evaluation/fevp_metric_contracts.py",
    "src/floorflow/evaluation/predicted_edit_graph_metrics.py",
    "src/floorflow/evaluation/structural_legality.py",
    "configs/fevp_metric_contract_v2.json",
    "configs/housemind_style_ged_deterministic_v2.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--operation-contract", type=Path, default=DEFAULT_OPS)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
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
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    return commit, dirty


def main() -> None:
    args = parse_args()
    commit, dirty = _git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "refusing to freeze from a dirty worktree; commit the selected model "
            "and evaluator, or explicitly pass --allow-dirty for a non-authoritative draft"
        )
    required = (
        args.checkpoint,
        args.config,
        args.prompt_manifest,
        args.taxonomy,
        args.operation_contract,
        args.heldout_manifest,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing freeze inputs: {missing}")
    payload = {
        "schema_version": "predicted-control-submission-freeze-v1",
        "status": "frozen",
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
        "evaluator_sha256": {
            name: _sha256(ROOT / name)
            for name in EVALUATOR_FILES
        },
        "heldout_manifest": str(args.heldout_manifest),
        "heldout_manifest_sha256": _sha256(args.heldout_manifest),
        "thresholds": {
            "operation_semantic_success": 0.9,
            "unsupported_abstention": 1.0,
            "outside_pixel_exact": 1.0,
            "outside_room_label_exact": 1.0,
            "validity": 1.0,
            "plan_plausibility": 0.95,
            "functional_watertightness": 0.95
        },
        "tolerances": {
            "room_match_min_iou": 0.5,
            "room_polygon_hausdorff_px": 0.5,
            "edge_contact_px": 2.0,
            "opening_center_px": 1.0,
            "changed_pixel_epsilon": 0.0
        },
        "freeze_policy": [
            "No checkpoint or threshold changes after held-out access.",
            "Held-out is executed once.",
            "Negative and failed metrics remain in the report."
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": "frozen"}, indent=2))


if __name__ == "__main__":
    main()

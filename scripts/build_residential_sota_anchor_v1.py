#!/usr/bin/env python3
"""Freeze the Raster2Seq fidelity anchor and residential evaluation boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT = "f7c0ef9379aa8cf11349da8b2179214e0d495af758383984a2361a1309aab3b7"
EDIT_MANIFESTS = (
    "outputs/control_contracts/fp_local_edit_room_type_test503_v1/test/manifest.json",
    "outputs/control_contracts/fp_local_edit_room_add_test503_v1/test/manifest.json",
    "outputs/control_contracts/fp_local_edit_room_remove_test503_v1/test/manifest.json",
    "outputs/control_contracts/fp_local_edit_opening_delete_test503_v1/test/manifest.json",
    "outputs/control_contracts/fp_local_edit_furniture_delete_test503_v1/test/manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "outputs/reports/fp_raster2seq_official_eval_metrics_v1.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/cubicasa5k/checkpoint.pth",
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=ROOT
        / ".data/external/raster2seq/coco_cubicasa5k_nowalls_v4_test_refined",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / ".data/cubicasa5k_source",
    )
    parser.add_argument(
        "--raster2seq-repo",
        type=Path,
        default=ROOT / "third_party/Raster2Seq",
    )
    parser.add_argument(
        "--anchor-output",
        type=Path,
        default=ROOT / "outputs/reports/residential_sota_anchor_v1.json",
    )
    parser.add_argument(
        "--lockbox-output",
        type=Path,
        default=ROOT / "configs/residential_sota_lockbox_v1.json",
    )
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")


def edit_lockbox() -> dict:
    rows = {}
    names = set()
    edited_rows = 0
    for relative in EDIT_MANIFESTS:
        path = ROOT / relative
        payload = read(path)
        selected = [row for row in payload["rows"] if row["edited"]]
        edited_rows += len(selected)
        names.update(str(row["name"]) for row in selected)
        rows[relative] = {
            "all_rows": len(payload["rows"]),
            "edited_rows": len(selected),
            "sha256": digest(path),
        }
    return {
        "manifests": rows,
        "edited_rows": edited_rows,
        "unique_edited_cases": len(names),
        "policy": "locked edit test set; evaluate only after adapter freeze",
    }


def official_lockbox(root: Path) -> dict:
    annotations = sorted((root / "annotations_json/test").glob("*.json"))
    images = sorted((root / "test").glob("*.png"))
    names = [path.stem for path in annotations]
    return {
        "root": str(root),
        "annotation_files": len(annotations),
        "image_files": len(images),
        "case_ids": names,
        "annotation_tree_sha256": hashlib.sha256(
            "\0".join(
                f"{path.stem}:{digest(path)}" for path in annotations
            ).encode("utf-8")
        ).hexdigest(),
        "policy": "locked official test509; evaluate once after checkpoint freeze",
    }


def source_lockbox(root: Path) -> dict:
    return {
        "root": str(root),
        "train_ids": sum(
            1 for line in (root / "train.txt").read_text().splitlines() if line
        ),
        "val_ids": sum(
            1 for line in (root / "val.txt").read_text().splitlines() if line
        ),
        "test_ids": sum(
            1 for line in (root / "test.txt").read_text().splitlines() if line
        ),
        "train_list_sha256": digest(root / "train.txt"),
        "val_list_sha256": digest(root / "val.txt"),
        "test_list_sha256": digest(root / "test.txt"),
        "policy": "only train/val may be used for adapter fitting or calibration",
    }


def main() -> None:
    args = parse_args()
    metrics = read(args.metrics)
    edited_status = edit_lockbox()
    official_status = official_lockbox(args.official_root)
    source_status = source_lockbox(args.source_root)
    modified = git_output(
        args.raster2seq_repo,
        "status",
        "--short",
        "--untracked-files=no",
    ).splitlines()
    modified_hashes = {
        relative: digest(args.raster2seq_repo / relative)
        for line in modified
        if (relative := line[3:].strip())
    }
    checkpoint_sha = digest(args.checkpoint)
    values = metrics["metrics"]
    checks = {
        "checkpoint_hash_matches_anchor": checkpoint_sha == EXPECTED_CHECKPOINT,
        "official_cases_509": metrics["cases"] == 509,
        "official_room_f1_reproduced": abs(
            float(values["room_f1"])
            - float(metrics["published_room_f1_anchor"])
        )
        <= 0.002,
        "official_test_locked": official_status["annotation_files"] == 509,
        "edit_rows_2479": edited_status["edited_rows"] == 2479,
        "unique_base_cases_502": edited_status["unique_edited_cases"] == 502,
        "train_ids_4200": source_status["train_ids"] == 4200,
        "val_ids_400": source_status["val_ids"] == 400,
        "test_ids_400": source_status["test_ids"] == 400,
    }
    report = {
        "status": "complete" if all(checks.values()) else "failed",
        "scope": "residential Raster2Seq fidelity anchor and evaluation lockboxes",
        "published_anchor": {
            "RoomF1": metrics["published_room_f1_anchor"],
        },
        "local_reproduction": {
            "cases": metrics["cases"],
            "metrics": values,
            "report": str(args.metrics.relative_to(ROOT)),
        },
        "checkpoint": {
            "path": str(args.checkpoint.relative_to(ROOT)),
            "sha256": checkpoint_sha,
        },
        "raster2seq_code": {
            "path": str(args.raster2seq_repo.relative_to(ROOT)),
            "commit": git_output(args.raster2seq_repo, "rev-parse", "HEAD"),
            "tracked_modified_files": modified_hashes,
        },
        "official_test": official_status,
        "edit_test": edited_status,
        "training_source": source_status,
        "checks": checks,
        "claim_boundary": (
            "Identity RoomF1 and editing metrics are separate protocols. "
            "No test set was used for fitting or hyperparameter selection."
        ),
    }
    lockbox = {
        "version": 1,
        "official_test": official_status,
        "edit_test": edited_status,
        "training_source": source_status,
        "checkpoint": report["checkpoint"],
        "evaluation_policy": {
            "fit_splits": ["train", "val"],
            "official_test_runs": 1,
            "edit_test_runs_after_freeze": 1,
            "no_best_of_n": True,
        },
    }
    args.anchor_output.parent.mkdir(parents=True, exist_ok=True)
    args.lockbox_output.parent.mkdir(parents=True, exist_ok=True)
    args.anchor_output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    args.lockbox_output.write_text(
        json.dumps(lockbox, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

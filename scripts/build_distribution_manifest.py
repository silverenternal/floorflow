#!/usr/bin/env python3
"""Freeze model, evaluation and provenance fields for a distribution run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distribution_adapter_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, default=None)
    parser.add_argument("--test-accessed", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_hashes() -> dict[str, str]:
    files = [
        "src/floorflow/distribution/cfm_model.py",
        "src/floorflow/distribution/element_head.py",
        "src/floorflow/distribution/hub.py",
        "src/floorflow/distribution/joint_loss.py",
        "src/floorflow/inference/distribution_overlay.py",
        "src/floorflow/flow/cfm_backend.py",
        "scripts/train_element_adapter.py",
        "scripts/evaluate_distribution_checkpoint.py",
        "scripts/evaluate_distribution_overlay.py",
    ]
    return {
        path: sha256(Path(path))
        for path in files
        if Path(path).is_file()
    }


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint = Path(config["checkpoint"]).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = payload.get("config", {})
    if model_config.get("element_mask_version") != config["element_mask_version"]:
        raise ValueError("checkpoint element-mask version does not match manifest")
    git_state = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    manifest = {
        "status": "frozen",
        "created_utc": datetime.now(UTC).isoformat(),
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "model_config": model_config,
        "training_args": jsonable(payload.get("training_args", {})),
        "condition_mode": "contour_only",
        "target_derived_control": False,
        "best_of_n_used": False,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchcfm": package_version("torchcfm"),
            "cuda": torch.version.cuda,
        },
        "source_sha256": source_hashes(),
        "evaluation_config": config["evaluation"],
        "selection": config["selection"],
        "seed": config["seed"],
        "git_status": git_state,
        "test_accessed": bool(args.test_accessed),
        "parent_manifest": (
            str(args.parent_manifest)
            if args.parent_manifest is not None
            else None
        ),
        "parent_manifest_sha256": (
            sha256(args.parent_manifest)
            if args.parent_manifest is not None
            else None
        ),
        "model_frozen_before_test": args.parent_manifest is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

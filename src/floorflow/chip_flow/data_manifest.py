"""Stable data fingerprints shared by chip training and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from floorflow.chip_flow.cache import ARRAY_FIELDS
from floorflow.chip_flow.training_runtime import write_json_atomic


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_fingerprint(root: str | Path) -> dict[str, object]:
    directory = Path(root)
    files = sorted(
        path for path in directory.iterdir() if path.suffix in {".json", ".npy"}
    )
    if not files:
        raise FileNotFoundError(f"cache directory has no arrays: {directory}")
    return {
        "root": str(directory.resolve()),
        "schema_fields": list(ARRAY_FIELDS),
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }


def build_data_manifest(train_root: str | Path, dev_root: str | Path) -> dict:
    return {
        "version": "chip-data-manifest-v1",
        "train": cache_fingerprint(train_root),
        "dev": cache_fingerprint(dev_root),
    }


def write_data_manifest(path: str | Path, manifest: dict) -> None:
    write_json_atomic(path, manifest)


def verify_data_manifest(
    manifest: dict,
    train_root: str | Path,
    dev_root: str | Path,
) -> None:
    if build_data_manifest(train_root, dev_root) != manifest:
        raise ValueError("data cache does not match the frozen data manifest")


def load_data_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "build_data_manifest",
    "cache_fingerprint",
    "load_data_manifest",
    "sha256_file",
    "verify_data_manifest",
    "write_data_manifest",
]

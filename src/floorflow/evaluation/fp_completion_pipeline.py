"""Shared loading and path helpers for the FP completion pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from floorflow.data.control_contract import CONTROL_SCHEMA_VERSION, sha256_file
from floorflow.heads.coarse_furniture_refiner import CoarseFurnitureRefiner
from floorflow.heads.coarse_layout_refiner import CoarseLayoutRefiner
from floorflow.heads.coarse_structure_refiner import CoarseStructureRefiner
from floorflow.heads.coarse_support_refiner import CoarseSupportRefiner


def load_completion_modules(
    room_checkpoint: Path,
    structure_checkpoint: Path,
    furniture_checkpoint: Path,
    support_checkpoint: Path,
    taxonomy,
    device: torch.device,
):
    """Load the four deterministic completion stages from their checkpoints."""

    room_payload = torch.load(
        room_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    room = CoarseLayoutRefiner(len(taxonomy.room_ids)).to(device)
    room.load_state_dict(room_payload["model"])
    room.eval()

    structure_payload = torch.load(
        structure_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    structure = CoarseStructureRefiner(
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        len(taxonomy.opening_ids),
        base_channels=int(structure_payload["config"].get("base_channels", 32)),
    ).to(device)
    structure.load_state_dict(structure_payload["model"])
    structure.eval()

    furniture_payload = torch.load(
        furniture_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    furniture = CoarseFurnitureRefiner(
        len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
        len(taxonomy.furniture_ids),
        base_channels=int(furniture_payload["config"].get("base_channels", 32)),
    ).to(device)
    furniture.load_state_dict(furniture_payload["model"])
    furniture.eval()

    support_payload = torch.load(
        support_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    support = CoarseSupportRefiner(
        len(taxonomy.room_ids) + 4,
        base_channels=int(support_payload["config"].get("base_channels", 32)),
    ).to(device)
    support.load_state_dict(support_payload["model"])
    support.eval()
    return room, structure, furniture, support


def load_control_manifest(
    *,
    control_root: Path | None,
    control_manifest: Path | None,
    split: str,
    dataset,
    limit: int | None,
) -> dict | None:
    """Load and validate the optional serialized-control manifest."""

    if control_root is None:
        return None
    path = (
        control_manifest
        if control_manifest is not None
        else control_root / split / "manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != CONTROL_SCHEMA_VERSION:
        raise ValueError(
            f"control schema {manifest['schema_version']} != {CONTROL_SCHEMA_VERSION}"
        )
    if manifest["split"] != split:
        raise ValueError(f"control split {manifest['split']} != {split}")
    manifest_cases = int(manifest["cases"])
    if limit is None and manifest_cases != len(dataset):
        raise ValueError(
            f"control cases {manifest_cases} != dataset cases {len(dataset)}"
        )
    if limit is not None and len(dataset) > manifest_cases:
        raise ValueError(
            f"dataset limit {len(dataset)} exceeds control cases {manifest_cases}"
        )
    manifest["manifest_path"] = str(path)
    manifest["manifest_sha256"] = sha256_file(path)
    return manifest


def geometry_path(root: Path, split: str, name: str) -> Path:
    floor = Path(name.replace("__", "/"))
    return (
        root
        / split
        / floor.parent
        / f"{floor.name.replace('-', '_')}.geometry.npz"
    )


def support_iou(prediction, target, region) -> float:
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


__all__ = [
    "geometry_path",
    "load_completion_modules",
    "load_control_manifest",
    "support_iou",
]

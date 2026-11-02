"""Serialized control contract for floorplan completion experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from floorflow.data.control_cache import read_control_cache
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.coarse_furniture_refiner import coarse_furniture_control
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.heads.coarse_support_refiner import coarse_support_control
from floorflow.heads.room_program import room_presence_target
from floorflow.representation.coarse_layout import coarse_room_layout_logits

CONTROL_SCHEMA_VERSION = "fp-control-v1"
CONTROL_SOURCES = {
    "oracle": {"deployable": False, "target_used_for_control": True},
    "predicted": {"deployable": True, "target_used_for_control": False},
    "external": {"deployable": True, "target_used_for_control": False},
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def oracle_control_payload(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    room_size: int = 224,
    structure_size: int = 128,
    furniture_size: int = 192,
    support_size: int = 192,
) -> dict[str, torch.Tensor]:
    """Build a control bundle from labels for an oracle upper bound."""

    if labels.ndim != 3 or footprint.ndim != 3:
        raise ValueError("labels and footprint must be [B,H,W]")
    if labels.shape != footprint.shape:
        raise ValueError("labels and footprint must share a shape")
    room_logits = coarse_room_layout_logits(
        labels,
        taxonomy,
        size=room_size,
    )
    room_logits = F.interpolate(
        room_logits,
        size=(int(room_size), int(room_size)),
        mode="bilinear",
        align_corners=False,
    )
    return {
        "room_logits": room_logits,
        "program": room_presence_target(labels, taxonomy).bool(),
        "structure": coarse_structure_control(
            labels,
            footprint,
            taxonomy,
            size=structure_size,
        ),
        "furniture": coarse_furniture_control(
            labels,
            footprint,
            taxonomy,
            size=furniture_size,
        ),
        "support": coarse_support_control(
            labels,
            footprint,
            taxonomy,
            size=support_size,
        ),
    }


def expected_control_shapes(
    taxonomy: FloorplanTaxonomy,
    *,
    room_size: int,
    structure_size: int,
    furniture_size: int,
    support_size: int,
) -> dict[str, tuple[int, int, int]]:
    room_count = len(taxonomy.room_ids)
    return {
        "room_logits": (room_count, room_size, room_size),
        "program": (room_count,),
        "structure": (
            room_count + len(taxonomy.opening_ids) + 3,
            structure_size,
            structure_size,
        ),
        "furniture": (
            room_count + len(taxonomy.furniture_ids) + 1,
            furniture_size,
            furniture_size,
        ),
        "support": (room_count + 4, support_size, support_size),
    }


def validate_control_payload(
    payload: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    room_size: int = 224,
    structure_size: int = 128,
    furniture_size: int = 192,
    support_size: int = 192,
) -> dict[str, object]:
    """Validate a serialized bundle and return stable per-array summaries."""

    expected = expected_control_shapes(
        taxonomy,
        room_size=room_size,
        structure_size=structure_size,
        furniture_size=furniture_size,
        support_size=support_size,
    )
    missing = sorted(set(expected) - set(payload))
    extra = sorted(set(payload) - set(expected))
    if missing or extra:
        raise ValueError(f"control arrays mismatch: missing={missing}, extra={extra}")

    summaries: dict[str, object] = {}
    for name, shape in expected.items():
        value = payload[name]
        if tuple(value.shape) != shape:
            raise ValueError(
                f"{name} has shape {tuple(value.shape)}, expected {shape}"
            )
        if name == "program":
            if value.dtype != torch.bool:
                raise ValueError("program must be boolean")
            summaries[name] = {
                "shape": list(shape),
                "dtype": str(value.dtype),
                "count": int(value.sum()),
            }
            continue
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        summaries[name] = {
            "shape": list(shape),
            "dtype": str(value.dtype),
            "count_nonzero": int(torch.count_nonzero(value)),
            "min": float(value.min()),
            "max": float(value.max()),
        }
    return summaries


def summarize_control_file(
    path: str | Path,
    taxonomy: FloorplanTaxonomy,
    *,
    room_size: int = 224,
    structure_size: int = 128,
    furniture_size: int = 192,
    support_size: int = 192,
) -> dict[str, object]:
    path = Path(path)
    payload = read_control_cache(path)
    arrays = validate_control_payload(
        payload,
        taxonomy,
        room_size=room_size,
        structure_size=structure_size,
        furniture_size=furniture_size,
        support_size=support_size,
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "arrays": arrays,
    }


def control_provenance(source: str) -> dict[str, bool]:
    if source not in CONTROL_SOURCES:
        raise ValueError(f"unknown control source: {source}")
    return {
        "target_used_for_control": CONTROL_SOURCES[source][
            "target_used_for_control"
        ],
        "deployable": CONTROL_SOURCES[source]["deployable"],
    }


__all__ = [
    "CONTROL_SCHEMA_VERSION",
    "CONTROL_SOURCES",
    "control_provenance",
    "expected_control_shapes",
    "oracle_control_payload",
    "sha256_file",
    "summarize_control_file",
    "validate_control_payload",
]

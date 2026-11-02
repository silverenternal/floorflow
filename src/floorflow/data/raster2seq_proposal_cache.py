"""Typed storage for structure-conditioned Raster2Seq proposal-teacher outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy

SCHEMA_VERSION = "raster2seq-proposal-cache-v1"
MANIFEST_VERSION = "raster2seq-proposal-cache-manifest-v1"


@dataclass(frozen=True)
class TeacherProposalCase:
    sample_key: str
    split: str
    split_index: int
    image_size: int
    polygons: tuple[np.ndarray, ...]
    class_ids: np.ndarray
    class_names: tuple[str, ...]
    existence_scores: np.ndarray
    masks: np.ndarray
    sequence_length: int
    source_image_sha256: str


def sha256_file(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _proposal_masks(
    polygons: list[np.ndarray],
    image_size: int,
) -> np.ndarray:
    masks = np.zeros((len(polygons), image_size, image_size), dtype=bool)
    for index, polygon in enumerate(polygons):
        local = masks[index].astype(np.uint8)
        cv2.fillPoly(
            local,
            [
                np.rint(
                    polygon * float(image_size - 1)
                ).astype(np.int32)
            ],
            1,
        )
        masks[index] = local.astype(bool)
    return masks


def write_teacher_case(
    path: str | Path,
    *,
    sample_key: str,
    split: str,
    split_index: int,
    image_size: int,
    polygons: list[np.ndarray],
    class_ids: list[int],
    class_names: list[str],
    existence_scores: list[float],
    sequence_length: int,
    source_image_sha256: str,
) -> Path:
    output = Path(path)
    count = len(polygons)
    if not (
        len(class_ids)
        == len(class_names)
        == len(existence_scores)
        == count
    ):
        raise ValueError("proposal fields must have equal lengths")
    normalized = []
    for polygon in polygons:
        array = np.asarray(polygon, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
            raise ValueError("room polygons must be [N>=3,2]")
        if np.any(array < 0.0) or np.any(array > float(image_size - 1)):
            raise ValueError("polygon coordinates are outside the image frame")
        normalized.append(array / float(image_size - 1))
    points = (
        np.concatenate(normalized, axis=0)
        if normalized
        else np.zeros((0, 2), dtype=np.float32)
    )
    offsets = np.zeros(count + 1, dtype=np.int32)
    if count:
        offsets[1:] = np.cumsum([len(polygon) for polygon in normalized])
    scores = np.asarray(existence_scores, dtype=np.float32)
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("existence scores must be finite in [0,1]")
    masks = _proposal_masks(normalized, image_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(SCHEMA_VERSION),
        sample_key=np.asarray(sample_key),
        split=np.asarray(split),
        split_index=np.int32(split_index),
        image_size=np.int32(image_size),
        polygon_points=points.astype(np.float32, copy=False),
        polygon_offsets=offsets,
        class_ids=np.asarray(class_ids, dtype=np.int64),
        class_names=np.asarray(class_names, dtype="<U64"),
        existence_scores=scores,
        proposal_masks=masks,
        sequence_length=np.int32(sequence_length),
        source_image_sha256=np.asarray(source_image_sha256),
    )
    return output


def read_teacher_case(
    path: str | Path,
    *,
    taxonomy: FloorplanTaxonomy | None = None,
) -> TeacherProposalCase:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "schema_version",
            "sample_key",
            "split",
            "split_index",
            "image_size",
            "polygon_points",
            "polygon_offsets",
            "class_ids",
            "class_names",
            "existence_scores",
            "proposal_masks",
            "sequence_length",
            "source_image_sha256",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"teacher cache is missing fields: {sorted(missing)}")
        schema_version = str(payload["schema_version"].item())
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported teacher cache schema: {schema_version}")
        sample_key = str(payload["sample_key"].item())
        split = str(payload["split"].item())
        split_index = int(payload["split_index"])
        image_size = int(payload["image_size"])
        points = payload["polygon_points"].astype(np.float32, copy=False)
        offsets = payload["polygon_offsets"].astype(np.int64, copy=False)
        class_ids = payload["class_ids"].astype(np.int64, copy=False)
        class_names = tuple(str(value) for value in payload["class_names"])
        scores = payload["existence_scores"].astype(np.float32, copy=False)
        masks = payload["proposal_masks"].astype(bool, copy=False)
        sequence_length = int(payload["sequence_length"])
        source_image_sha256 = str(payload["source_image_sha256"].item())
    count = len(class_ids)
    if offsets.shape != (count + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 3):
        raise ValueError("invalid polygon offsets")
    if offsets[-1] != len(points):
        raise ValueError("polygon offsets do not cover polygon points")
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError("invalid polygon point shape")
    if np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError("normalized polygon points must be in [0,1]")
    if scores.shape != (count,) or masks.shape != (count, image_size, image_size):
        raise ValueError("proposal masks or existence scores have invalid shapes")
    if len(class_names) != count:
        raise ValueError("class name count does not match proposal count")
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("invalid existence scores")
    if taxonomy is not None:
        allowed = {
            taxonomy.class_id(name): name
            for name in taxonomy.class_names
            if name in class_names
        }
        if any(
            int(class_id) not in allowed or allowed[int(class_id)] != name
            for class_id, name in zip(class_ids, class_names, strict=True)
        ):
            raise ValueError("teacher taxonomy does not match the supplied taxonomy")
    polygons = tuple(
        points[int(offsets[index]) : int(offsets[index + 1])]
        for index in range(count)
    )
    return TeacherProposalCase(
        sample_key=sample_key,
        split=split,
        split_index=split_index,
        image_size=image_size,
        polygons=polygons,
        class_ids=class_ids.copy(),
        class_names=class_names,
        existence_scores=scores.copy(),
        masks=masks.copy(),
        sequence_length=sequence_length,
        source_image_sha256=source_image_sha256,
    )


def read_manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_VERSION:
        raise ValueError("unsupported teacher cache manifest schema")
    if payload.get("target_used_at_inference") is not False:
        raise ValueError("teacher manifest must declare target-free inference")
    keys = [str(row["sample_key"]) for row in payload.get("rows", [])]
    if len(keys) != len(set(keys)):
        raise ValueError("teacher manifest contains duplicate sample keys")
    return payload


def iter_teacher_cases(
    manifest_path: str | Path,
    *,
    taxonomy: FloorplanTaxonomy | None = None,
) -> Iterator[TeacherProposalCase]:
    manifest = read_manifest(manifest_path)
    root = Path(manifest_path).parent
    for row in manifest["rows"]:
        yield read_teacher_case(root / str(row["cache_file"]), taxonomy=taxonomy)


__all__ = [
    "MANIFEST_VERSION",
    "SCHEMA_VERSION",
    "TeacherProposalCase",
    "iter_teacher_cases",
    "read_manifest",
    "read_teacher_case",
    "sha256_file",
    "write_teacher_case",
]

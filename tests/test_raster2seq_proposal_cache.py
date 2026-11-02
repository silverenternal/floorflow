from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from floorflow.data.raster2seq_proposal_cache import (
    MANIFEST_VERSION,
    read_manifest,
    read_teacher_case,
    write_teacher_case,
)
from floorflow.data.taxonomy import FloorplanTaxonomy


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=("background", "wall", "room:LivingRoom", "room:Bedroom"),
        room_map={"LivingRoom": "room:LivingRoom", "Bedroom": "room:Bedroom"},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_teacher_cache_round_trip_preserves_proposal_interface(tmp_path: Path) -> None:
    taxonomy = _taxonomy()
    cache_path = tmp_path / "case.npz"
    write_teacher_case(
        cache_path,
        sample_key="case__Floor-1",
        split="val",
        split_index=7,
        image_size=32,
        polygons=[
            np.asarray([[2, 2], [2, 14], [14, 14], [14, 2]], dtype=np.float32),
            np.asarray([[17, 2], [17, 14], [29, 14], [29, 2]], dtype=np.float32),
        ],
        class_ids=[
            taxonomy.class_id("room:LivingRoom"),
            taxonomy.class_id("room:Bedroom"),
        ],
        class_names=["room:LivingRoom", "room:Bedroom"],
        existence_scores=[0.8, 0.7],
        sequence_length=42,
        source_image_sha256="a" * 64,
    )
    case = read_teacher_case(cache_path, taxonomy=taxonomy)
    assert case.sample_key == "case__Floor-1"
    assert case.split == "val"
    assert case.split_index == 7
    assert len(case.polygons) == 2
    assert case.class_names == ("room:LivingRoom", "room:Bedroom")
    assert np.allclose(case.existence_scores, [0.8, 0.7])
    assert case.masks.shape == (2, 32, 32)
    assert case.masks[0, 5, 5]
    assert case.masks[1, 5, 20]
    assert not case.masks[0, 5, 20]


def test_teacher_cache_rejects_taxonomy_mismatch(tmp_path: Path) -> None:
    cache_path = tmp_path / "case.npz"
    write_teacher_case(
        cache_path,
        sample_key="case__Floor-1",
        split="train",
        split_index=0,
        image_size=16,
        polygons=[np.asarray([[1, 1], [1, 8], [8, 8]], dtype=np.float32)],
        class_ids=[2],
        class_names=["room:Bedroom"],
        existence_scores=[0.5],
        sequence_length=4,
        source_image_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="taxonomy"):
        read_teacher_case(cache_path, taxonomy=_taxonomy())


def test_teacher_manifest_enforces_target_free_provenance(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = {
        "schema_version": MANIFEST_VERSION,
        "target_used_at_inference": True,
        "rows": [{"sample_key": "case__Floor-1", "cache_file": "cases/case.npz"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="target-free"):
        read_manifest(path)

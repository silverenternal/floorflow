from __future__ import annotations

import json

import pytest
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.training.cad_wall_candidate_epoch_cache import (
    _cache_fingerprint,
    cache_size,
    load_candidate_batch,
    prepare_candidate_cache,
    read_candidate_batch,
    write_candidate_batch,
)
from floorflow.training.cad_wall_candidate_training import CandidateBatch


def test_candidate_epoch_cache_reuses_frozen_layer_output() -> None:
    taxonomy = build_taxonomy(
        {
            "room": {},
            "furniture": {},
            "door": {},
            "window": {},
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    calls = {"controls": 0}

    def controls_loader(_root, _split, _names, _args, _device):
        calls["controls"] += 1
        return {"opening_family_probability": torch.ones(1, 1, 4, 4)}

    def stage_runtime(controls, footprint, _image, _floor_ids):
        return controls, {}

    def layer_model(_controls, _footprint):
        return {
            "action_tokens": {"rooms": torch.ones(1, 3)},
            "field_features": torch.ones(1, 3, 4, 4),
        }

    candidate = CandidateBatch(
        endpoints=torch.zeros(1, 2, 2, 2),
        valid=torch.tensor([[True, False]]),
        evidence=torch.tensor([[0.5, 0.0]]),
        exterior=torch.tensor([[True, False]]),
        rooms=[[{"polygon": []}]],
        host_walls=[[]],
    )

    def candidate_builder(*_args, **_kwargs):
        return candidate

    cache = {}
    kwargs = {
        "cache": cache,
        "split": "train",
        "names": ["case"],
        "root": None,
        "args": None,
        "device": torch.device("cpu"),
        "footprint": torch.ones(1, 4, 4),
        "image": torch.zeros(1, 1, 4, 4),
        "floor_ids": torch.zeros(1, dtype=torch.long),
        "taxonomy": taxonomy,
        "stage_runtime": stage_runtime,
        "layer_model": layer_model,
        "controls_loader": controls_loader,
        "graph_stages": ("rooms",),
        "minimum_room_area": 4,
        "max_walls": 2,
        "candidate_builder": candidate_builder,
    }
    first = load_candidate_batch(**kwargs)
    second = load_candidate_batch(**kwargs)
    assert calls["controls"] == 1
    assert torch.equal(first[0].endpoints, candidate.endpoints)
    assert torch.equal(second[0].endpoints, candidate.endpoints)
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert cache_size(cache) == 1


def test_candidate_disk_cache_roundtrip(tmp_path) -> None:
    candidates = CandidateBatch(
        endpoints=torch.zeros(1, 2, 2, 2),
        valid=torch.tensor([[True, False]]),
        evidence=torch.tensor([[0.5, 0.0]]),
        exterior=torch.tensor([[True, False]]),
        rooms=[[{"polygon": []}]],
        host_walls=[[]],
    )
    result = (
        candidates,
        torch.ones(1, 3),
        torch.ones(1, 1, 4, 4),
        torch.ones(1, 3, 4, 4),
    )
    write_candidate_batch(tmp_path, "train", ["case"], result)
    loaded = read_candidate_batch(
        tmp_path,
        "train",
        ["case"],
        torch.device("cpu"),
    )
    assert loaded is not None
    assert loaded[0].rooms == candidates.rooms
    assert torch.equal(loaded[1], result[1])
    assert torch.equal(loaded[2], result[2])
    assert torch.equal(loaded[3], result[3])


def test_candidate_cache_fingerprint_binds_control_manifests(tmp_path) -> None:
    predicted = tmp_path / "controls"
    for split in ("train", "val"):
        (predicted / split).mkdir(parents=True)
        (predicted / split / "manifest.json").write_text(
            json.dumps({"split": split, "cases": 1}),
            encoding="utf-8",
        )
    checkpoint = tmp_path / "layer.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = type(
        "Args",
        (),
        {
            "predicted_root": predicted,
            "cache_root": tmp_path / "cache",
            "geometry_root": tmp_path / "geometry",
            "data_root": tmp_path / "data",
            "minimum_room_area": 64,
            "max_walls": 128,
        },
    )()
    cache_root = tmp_path / "candidates"
    fingerprint = prepare_candidate_cache(cache_root, args, checkpoint)
    assert prepare_candidate_cache(cache_root, args, checkpoint) == fingerprint
    args.candidate_mode = "host"
    host_fingerprint = prepare_candidate_cache(
        tmp_path / "candidates-host-v7",
        args,
        checkpoint,
    )
    assert host_fingerprint != fingerprint
    args.candidate_mode = "boundary"
    (predicted / "train" / "manifest.json").write_text(
        json.dumps({"split": "train", "cases": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        prepare_candidate_cache(cache_root, args, checkpoint)


def test_legacy_candidate_cache_manifest_migrates_only_on_exact_match(
    tmp_path,
) -> None:
    predicted = tmp_path / "controls"
    predicted.mkdir()
    checkpoint = tmp_path / "layer.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = type(
        "Args",
        (),
        {
            "predicted_root": predicted,
            "cache_root": tmp_path / "cache",
            "geometry_root": tmp_path / "geometry",
            "data_root": tmp_path / "data",
            "minimum_room_area": 64,
            "max_walls": 128,
        },
    )()
    cache_root = tmp_path / "candidates"
    cache_root.mkdir()
    legacy = _cache_fingerprint(
        {
            "layer_checkpoint": str(checkpoint),
            "layer_size": checkpoint.stat().st_size,
            "layer_mtime_ns": checkpoint.stat().st_mtime_ns,
            "predicted_root": str(args.predicted_root),
            "cache_root": str(args.cache_root),
            "geometry_root": str(args.geometry_root),
            "data_root": str(args.data_root),
            "image_size": None,
            "minimum_room_area": 64,
            "max_walls": 128,
        }
    )
    manifest_path = cache_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cad-wall-candidate-cache-v1",
                "fingerprint": legacy,
            }
        ),
        encoding="utf-8",
    )
    prepare_candidate_cache(cache_root, args, checkpoint)
    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == "cad-wall-candidate-cache-v5"
    assert migrated["migrated_from_legacy_fingerprint"] == legacy


def test_transitional_v1_schema_with_current_fingerprint_upgrades(
    tmp_path,
) -> None:
    predicted = tmp_path / "controls"
    predicted.mkdir()
    checkpoint = tmp_path / "layer.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = type(
        "Args",
        (),
        {
            "predicted_root": predicted,
            "cache_root": tmp_path / "cache",
            "geometry_root": tmp_path / "geometry",
            "data_root": tmp_path / "data",
            "minimum_room_area": 64,
            "max_walls": 128,
        },
    )()
    cache_root = tmp_path / "candidates"
    first = prepare_candidate_cache(cache_root, args, checkpoint)
    manifest_path = cache_root / "manifest.json"
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cad-wall-candidate-cache-v1",
                "fingerprint": current["fingerprint"],
            }
        ),
        encoding="utf-8",
    )
    assert prepare_candidate_cache(cache_root, args, checkpoint) == first
    upgraded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert upgraded["schema_version"] == "cad-wall-candidate-cache-v5"

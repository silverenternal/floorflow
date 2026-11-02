"""In-memory cache for frozen wall-candidate training batches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.training.cad_wall_candidate_training import (
    CandidateBatch,
    build_candidate_batch,
)


def load_candidate_batch(
    *,
    cache: dict[tuple[str, str], dict[str, Any]],
    split: str,
    names: Sequence[str],
    root,
    args,
    device: torch.device,
    footprint: torch.Tensor,
    image: torch.Tensor,
    floor_ids: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    stage_runtime,
    layer_model,
    controls_loader: Callable,
    graph_stages: Sequence[str],
    minimum_room_area: int,
    max_walls: int,
    candidate_builder: Callable[..., CandidateBatch] = build_candidate_batch,
    disk_root: str | Path | None = None,
) -> tuple[CandidateBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reuse frozen layer/candidate computation while targets stay uncached."""

    keys = [(split, str(name)) for name in names]
    entries = [cache.get(key) for key in keys]
    if any(entry is None for entry in entries):
        if disk_root is not None:
            disk_entries = [
                read_candidate_case(disk_root, split, name)
                for name in names
            ]
            if all(entry is not None for entry in disk_entries):
                for key, entry in zip(keys, disk_entries, strict=True):
                    cache[key] = entry
                return _assemble_batch(disk_entries, device)
        controls = controls_loader(root, split, names, args, device)
        controls, _ = stage_runtime(
            controls,
            footprint,
            image.to(device),
            floor_ids.to(device),
        )
        with torch.no_grad():
            output = layer_model(controls, footprint)
            tokens = torch.stack(
                [output["action_tokens"][name] for name in graph_stages],
                dim=1,
            )
            field_features = output["field_features"]
        candidates = candidate_builder(
            controls,
            output,
            footprint,
            taxonomy,
            minimum_room_area=int(minimum_room_area),
            max_walls=int(max_walls),
            device=device,
            candidate_mode=getattr(args, "candidate_mode", "boundary"),
            visible_wall_boundary_width=int(
                getattr(args, "wall_boundary_width", 7)
            ),
        )
        opening_probability = F.interpolate(
            controls["opening_family_probability"].float(),
            size=tuple(footprint.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        result = (candidates, tokens, opening_probability, field_features)
        for index, key in enumerate(keys):
            entry = _as_case_entry(result, index)
            cache[key] = entry
            if disk_root is not None:
                write_candidate_case(
                    disk_root,
                    split,
                    names[index],
                    entry,
                )
        return result
    return _assemble_batch(entries, device)


def cache_size(cache: dict[tuple[str, str], Any]) -> int:
    """Expose the number of cached cases for run reports."""

    return len(cache)


def _cache_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_candidate_fingerprint(checkpoint: Path, args) -> str:
    return _cache_fingerprint(
        {
            "layer_checkpoint": str(checkpoint),
            "layer_size": checkpoint.stat().st_size,
            "layer_mtime_ns": checkpoint.stat().st_mtime_ns,
            "predicted_root": str(args.predicted_root),
            "cache_root": str(args.cache_root),
            "geometry_root": str(args.geometry_root),
            "data_root": str(args.data_root),
            "image_size": getattr(args, "image_size", None),
            "minimum_room_area": int(args.minimum_room_area),
            "max_walls": int(args.max_walls),
        }
    )


def _candidate_fingerprint(
    checkpoint: Path,
    predicted_root: Path,
    args,
    contract_version: str,
    candidate_mode: str = "boundary",
) -> str:
    payload = {
            "candidate_contract_version": contract_version,
            "layer_checkpoint": str(checkpoint),
            "layer_checkpoint_sha256": _file_sha256(checkpoint),
            "control_manifests": {
                split: _file_sha256(predicted_root / split / "manifest.json")
                for split in ("train", "val")
            },
            "predicted_root": str(predicted_root),
            "cache_root": str(args.cache_root),
            "geometry_root": str(args.geometry_root),
            "data_root": str(args.data_root),
            "image_size": getattr(args, "image_size", None),
            "minimum_room_area": int(args.minimum_room_area),
            "max_walls": int(args.max_walls),
            "wall_boundary_width": int(
                getattr(args, "wall_boundary_width", 7)
            ),
            "room_program_bias": float(getattr(args, "room_program_bias", 0.0)),
        }
    if candidate_mode != "boundary":
        payload["candidate_mode"] = candidate_mode
    return _cache_fingerprint(payload)


def prepare_candidate_cache(
    root: str | Path,
    args,
    layer_checkpoint: str | Path,
) -> str:
    """Create or validate the frozen candidate cache manifest."""

    root = Path(root)
    checkpoint = Path(layer_checkpoint)
    predicted_root = Path(args.predicted_root)
    candidate_mode = str(getattr(args, "candidate_mode", "boundary"))
    contract_version = {
        "host": "cad-wall-candidate-v8-host-walls-geometry-junction",
        "skeleton_portfolio": (
            "cad-wall-candidate-v9-final-stage-skeleton-portfolio"
        ),
    }.get(
        candidate_mode,
        "cad-wall-candidate-v8-boundary-only-geometry-junction",
    )
    fingerprint = _candidate_fingerprint(
        checkpoint,
        predicted_root,
        args,
        contract_version,
        candidate_mode=candidate_mode,
    )
    path = root / "manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        existing = manifest.get("fingerprint")
        if existing == fingerprint:
            if manifest.get("schema_version") != "cad-wall-candidate-cache-v5":
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "cad-wall-candidate-cache-v5",
                            "fingerprint": fingerprint,
                            "upgraded_from_schema": manifest.get("schema_version"),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            return fingerprint
        if manifest.get("schema_version") == "cad-wall-candidate-cache-v1":
            legacy = _legacy_candidate_fingerprint(checkpoint, args)
            stage_is_default = (
                not bool(getattr(args, "use_junction_stage", False))
                and float(getattr(args, "room_program_bias", 0.0)) == 0.0
            )
            if existing != legacy or not stage_is_default:
                raise ValueError(
                    "wall-candidate cache fingerprint mismatch; use a new root"
                )
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "cad-wall-candidate-cache-v5",
                        "fingerprint": fingerprint,
                        "migrated_from_legacy_fingerprint": legacy,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        elif manifest.get("schema_version") == "cad-wall-candidate-cache-v2":
            boundary_v2 = _candidate_fingerprint(
                checkpoint,
                predicted_root,
                args,
                "cad-wall-candidate-v2",
            )
            if existing == boundary_v2:
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "cad-wall-candidate-cache-v5",
                            "fingerprint": fingerprint,
                            "migrated_from_boundary_fingerprint": boundary_v2,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                raise ValueError(
                    "wall-candidate cache fingerprint mismatch; use a new root"
                )
        else:
            raise ValueError(
                "wall-candidate cache fingerprint mismatch; use a new root"
            )
        return fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "cad-wall-candidate-cache-v5",
                "fingerprint": fingerprint,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return fingerprint


def candidate_cache_path(
    root: str | Path,
    split: str,
    name: str,
) -> Path:
    key = hashlib.sha256(str(name).encode("utf-8")).hexdigest()
    return Path(root) / split / f"{key}.pt"


def _cpu_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def write_candidate_batch(
    root: str | Path,
    split: str,
    names: Sequence[str],
    result,
) -> None:
    for index, name in enumerate(names):
        write_candidate_case(
            root,
            split,
            name,
            _as_case_entry(result, index),
        )


def write_candidate_case(
    root: str | Path,
    split: str,
    name: str,
    entry: dict[str, Any],
) -> None:
    """Persist one frozen candidate case for shuffle- and seed-independent reuse."""

    path = candidate_cache_path(root, split, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{path.stem}")
    torch.save(
        {key: _cpu_value(value) for key, value in entry.items()},
        temporary,
    )
    temporary.replace(path)


def _device_value(value, device: torch.device):
    return value.to(device) if isinstance(value, torch.Tensor) else value


def read_candidate_batch(
    root: str | Path,
    split: str,
    names: Sequence[str],
    device: torch.device,
):
    entries = [read_candidate_case(root, split, name) for name in names]
    if any(entry is None for entry in entries):
        return None
    return _assemble_batch(entries, device)


def _as_case_entry(result, index: int) -> dict[str, Any]:
    candidates, tokens, opening_probability, field_features = result
    control_endpoints = (
        candidates.control_endpoints
        if candidates.control_endpoints is not None
        else candidates.endpoints
    )
    control_valid = (
        candidates.control_valid
        if candidates.control_valid is not None
        else candidates.valid
    )
    control_evidence = (
        candidates.control_evidence
        if candidates.control_evidence is not None
        else candidates.evidence
    )
    return {
        "endpoints": candidates.endpoints[index],
        "valid": candidates.valid[index],
        "evidence": candidates.evidence[index],
        "control_endpoints": control_endpoints[index],
        "control_valid": control_valid[index],
        "control_evidence": control_evidence[index],
        "proposal_source": (
            candidates.proposal_source[index]
            if candidates.proposal_source is not None
            else candidates.valid[index].long()
        ),
        "control_proposal_source": (
            candidates.control_proposal_source[index]
            if candidates.control_proposal_source is not None
            else control_valid[index].long()
        ),
        "exterior": candidates.exterior[index],
        "rooms": candidates.rooms[index],
        "host_walls": candidates.host_walls[index],
        "tokens": tokens[index],
        "opening_probability": opening_probability[index],
        "field_features": field_features[index],
    }


def read_candidate_case(
    root: str | Path,
    split: str,
    name: str,
) -> dict[str, Any] | None:
    path = candidate_cache_path(root, split, name)
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _assemble_batch(entries, device: torch.device):
    candidates = CandidateBatch(
        endpoints=torch.stack(
            [_device_value(entry["endpoints"], device) for entry in entries]
        ),
        valid=torch.stack(
            [_device_value(entry["valid"], device) for entry in entries]
        ),
        evidence=torch.stack(
            [_device_value(entry["evidence"], device) for entry in entries]
        ),
        exterior=torch.stack(
            [_device_value(entry["exterior"], device) for entry in entries]
        ),
        rooms=[entry["rooms"] for entry in entries],
        host_walls=[entry["host_walls"] for entry in entries],
        control_endpoints=torch.stack(
            [
                _device_value(entry.get("control_endpoints", entry["endpoints"]), device)
                for entry in entries
            ]
        ),
        control_valid=torch.stack(
            [
                _device_value(entry.get("control_valid", entry["valid"]), device)
                for entry in entries
            ]
        ),
        control_evidence=torch.stack(
            [
                _device_value(
                    entry.get("control_evidence", entry["evidence"]),
                    device,
                )
                for entry in entries
            ]
        ),
        proposal_source=torch.stack(
            [
                _device_value(
                    entry.get("proposal_source", torch.zeros_like(entry["valid"])),
                    device,
                )
                for entry in entries
            ]
        ),
        control_proposal_source=torch.stack(
            [
                _device_value(
                    entry.get(
                        "control_proposal_source",
                        torch.zeros_like(entry["valid"]),
                    ),
                    device,
                )
                for entry in entries
            ]
        ),
    )
    return (
        candidates,
        torch.stack(
            [_device_value(entry["tokens"], device) for entry in entries]
        ),
        torch.stack(
            [
                _device_value(entry["opening_probability"], device)
                for entry in entries
            ]
        ),
        torch.stack(
            [
                _device_value(entry["field_features"], device)
                for entry in entries
            ]
        ),
    )


__all__ = [
    "cache_size",
    "candidate_cache_path",
    "load_candidate_batch",
    "prepare_candidate_cache",
    "read_candidate_batch",
    "read_candidate_case",
    "write_candidate_batch",
    "write_candidate_case",
]

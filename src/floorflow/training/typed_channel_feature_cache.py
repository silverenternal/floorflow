"""Disk cache for frozen predicted controls and field features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CONTROL_KEYS = (
    "room_logits",
    "structure",
    "furniture",
    "support",
    "opening_family_probability",
    "opening_foreground_probability",
    "opening_family_logits",
    "junction_probability",
)
FEATURE_KEYS = ("field_features", "evidence", "source_tokens")
CONTROL_CONTRACT = (
    "typed-channel-frozen-features-v2-independent-opening-junction"
)


def feature_cache_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_feature_cache_manifest(
    root: str | Path,
    fingerprint: str,
) -> None:
    path = Path(root) / "manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if str(manifest.get("fingerprint")) != str(fingerprint):
            raise ValueError(
                "feature cache fingerprint mismatch; use a new cache root"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": CONTROL_CONTRACT,
                "fingerprint": str(fingerprint),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def typed_channel_cache_fingerprint(args, model_config: dict) -> str:
    sources = {}
    for name in (
        "predicted_root",
        "room_checkpoint",
        "structure_checkpoint",
        "centerline_checkpoint",
        "opening_checkpoint",
    ):
        value = getattr(args, name, None)
        path = Path(value) if value is not None else None
        sources[name] = (
            {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            if path is not None and path.exists()
            else str(path)
        )
    return feature_cache_fingerprint(
        {
            "image_size": int(args.image_size),
            "room_blend": float(args.room_blend),
            "control_contract": CONTROL_CONTRACT,
            "model": model_config,
            "sources": sources,
        }
    )


def feature_cache_path(
    root: str | Path,
    split: str,
    name: str,
) -> Path:
    return Path(root) / split / f"{str(name).replace('/', '__')}.npz"


def extract_frozen_features(
    model: torch.nn.Module,
    controls: dict[str, torch.Tensor],
    footprint: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        features, evidence, sources = model.field.encode_details(
            controls,
            footprint,
        )
    output = {
        "field_features": features.detach().float(),
        "source_tokens": sources.detach().float(),
    }
    if evidence is not None:
        output["evidence"] = evidence.detach().float()
    return output


def write_frozen_batch(
    root: str | Path,
    split: str,
    names,
    controls: dict[str, torch.Tensor],
    features: dict[str, torch.Tensor],
) -> None:
    for row, name in enumerate(names):
        arrays = {
            key: controls[key][row].detach().cpu().numpy().astype(np.float32)
            for key in CONTROL_KEYS
            if key in controls
        }
        arrays["program"] = controls["program"][row].detach().bool().cpu().numpy()
        for key in FEATURE_KEYS:
            if key in features:
                arrays[key] = (
                    features[key][row].detach().cpu().numpy().astype(np.float16)
                )
        path = feature_cache_path(root, split, str(name))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)


def read_frozen_batch(
    root: str | Path,
    split: str,
    names,
    device: torch.device | str,
    *,
    non_blocking: bool | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None:
    controls = {key: [] for key in (*CONTROL_KEYS, "program")}
    features = {key: [] for key in FEATURE_KEYS}
    for name in names:
        path = feature_cache_path(root, split, str(name))
        if not path.is_file():
            return None
        with np.load(path) as payload:
            for key, values in controls.items():
                if key in payload:
                    value = torch.from_numpy(
                        payload[key].astype(np.bool_)
                        if key == "program"
                        else payload[key].astype(np.float32)
                    )
                    values.append(value)
            for key, values in features.items():
                if key in payload:
                    values.append(
                        torch.from_numpy(payload[key].astype(np.float32))
                    )
    resolved_device = torch.device(device)
    async_copy = (
        resolved_device.type == "cuda"
        if non_blocking is None
        else bool(non_blocking)
    )
    resolved_controls = {
        key: torch.stack(values).to(device, non_blocking=async_copy)
        for key, values in controls.items()
        if values
    }
    resolved_features = {
        key: torch.stack(values).to(device, non_blocking=async_copy)
        for key, values in features.items()
        if values
    }
    return resolved_controls, resolved_features


__all__ = [
    "CONTROL_KEYS",
    "FEATURE_KEYS",
    "ensure_feature_cache_manifest",
    "extract_frozen_features",
    "feature_cache_fingerprint",
    "feature_cache_path",
    "read_frozen_batch",
    "typed_channel_cache_fingerprint",
    "write_frozen_batch",
]

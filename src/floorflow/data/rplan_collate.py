"""Dynamic collation for RPLAN boundary and token targets."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

_VARIABLE_KEYS = {
    "token_ids",
    "token_mask",
    "token_loss_mask",
    "token_stage_ids",
    "token_kind_ids",
    "token_grammar_mask",
}


def _verifier_target_key(item: dict[str, Any]) -> str:
    """Return a deterministic target fingerprint without touching accelerator memory."""

    digest = hashlib.blake2b(digest_size=12)
    digest.update(str(item["case_name"]).encode("utf-8"))
    for key in (
        "token_ids",
        "token_mask",
        "token_stage_ids",
        "token_kind_ids",
        "room_exists",
        "source_room_count",
        "room_classes",
        "adjacency",
        "room_polygons",
        "room_point_valid",
        "room_target_labels",
        "room_target_mask",
        "boundary_points",
        "boundary_valid",
        "wall_endpoints",
        "wall_valid",
        "wall_exterior",
        "opening_center",
        "opening_width",
        "opening_host",
        "opening_valid",
        "junction_points",
        "junction_degree",
        "junction_valid",
    ):
        value = item.get(key)
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().to("cpu", copy=False).contiguous()
        digest.update(key.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _padded_token_tensor(
    items: list[dict[str, Any]],
    key: str,
    maximum: int,
) -> torch.Tensor:
    if key == "token_ids":
        padded = torch.full(
            (len(items), maximum),
            -1,
            dtype=torch.long,
        )
    elif key in {"token_stage_ids", "token_kind_ids"}:
        padded = torch.zeros((len(items), maximum), dtype=torch.long)
    else:
        padded = torch.zeros((len(items), maximum), dtype=torch.bool)
    for row, item in enumerate(items):
        value = item[key]
        padded[row, : value.shape[0]] = value
    return padded


def collate_rplan_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length token targets while preserving typed metadata."""

    if not items:
        raise ValueError("cannot collate an empty RPLAN batch")
    output: dict[str, Any] = {
        "case_name": [str(item["case_name"]) for item in items],
        "verifier_target_keys": [
            _verifier_target_key(item) for item in items
        ],
    }
    for output_key, source_key in (
        ("target_room_counts", "room_exists"),
        ("wall_target_counts", "wall_valid"),
        ("opening_target_counts", "opening_valid"),
    ):
        if all(source_key in item for item in items):
            output[output_key] = [
                int(item[source_key].sum().item())
                for item in items
            ]
    for key in items[0]:
        if key == "case_name":
            continue
        if key == "token_grammar_mask":
            continue
        if any(key not in item for item in items[1:]):
            raise ValueError(f"RPLAN item key presence mismatch: {key}")
        if key not in _VARIABLE_KEYS:
            output[key] = torch.stack([item[key] for item in items])
            continue
        maximum = max(int(item[key].shape[0]) for item in items)
        output[key] = _padded_token_tensor(items, key, maximum)
    grammar_presence = [
        "token_grammar_mask" in item
        for item in items
    ]
    if any(grammar_presence) and not all(grammar_presence):
        raise ValueError("token grammar mask presence mismatch")
    if all(grammar_presence):
        maximum = max(
            int(item["token_grammar_mask"].shape[0]) for item in items
        )
        vocab_sizes = {
            int(item["token_grammar_mask"].shape[1])
            for item in items
        }
        if len(vocab_sizes) != 1:
            raise ValueError("token grammar vocabulary size mismatch")
        vocab_size = vocab_sizes.pop()
        grammar = torch.zeros(
            (len(items), maximum, vocab_size),
            dtype=torch.bool,
        )
        for row, item in enumerate(items):
            value = item["token_grammar_mask"]
            grammar[row, : value.shape[0]] = value
        output["token_grammar_mask"] = grammar
    return output


__all__ = ["collate_rplan_items"]

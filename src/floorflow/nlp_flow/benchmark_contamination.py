"""Exact and shingle-based benchmark-vs-training contamination audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def text_fields(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        output: list[str] = []
        for item in value.values():
            output.extend(text_fields(item))
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(text_fields(item))
        return output
    return []


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def shingles(value: str, width: int) -> set[str]:
    normalized = normalize_text(value)
    if not normalized:
        return set()
    if len(normalized) <= width:
        return {digest(normalized)}
    return {
        digest(normalized[index : index + width])
        for index in range(0, len(normalized) - width + 1, max(1, width // 4))
    }


def index_training(paths: list[Path], width: int) -> tuple[set[str], set[str], int]:
    documents: set[str] = set()
    pieces: set[str] = set()
    rows = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                text = normalize_text(str(payload["text"]))
                documents.add(digest(text))
                pieces.update(shingles(text, width))
                rows += 1
    return documents, pieces, rows


def scan_benchmark(
    path: Path,
    training_documents: set[str],
    training_shingles: set[str],
    width: int,
    near_threshold: float,
) -> dict[str, object]:
    exact_matches = []
    near_matches = []
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            content = " ".join(text_fields(payload.get("fields", payload)))
            normalized = normalize_text(content)
            row_id = str(payload["id"])
            if digest(normalized) in training_documents:
                exact_matches.append(row_id)
            local = shingles(content, width)
            overlap = len(local & training_shingles) / max(1, len(local))
            if overlap >= near_threshold:
                near_matches.append({"id": row_id, "overlap": overlap})
            rows += 1
    return {
        "rows": rows,
        "exact_matches": exact_matches,
        "near_matches": near_matches,
    }


__all__ = [
    "index_training",
    "normalize_text",
    "scan_benchmark",
    "shingles",
    "text_fields",
]

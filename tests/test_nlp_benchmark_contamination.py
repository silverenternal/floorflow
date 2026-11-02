from __future__ import annotations

import json
from pathlib import Path

from floorflow.nlp_flow.benchmark_contamination import (
    index_training,
    normalize_text,
    scan_benchmark,
    shingles,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_contamination_scanner_detects_exact_and_near_overlap(tmp_path: Path) -> None:
    shared = "def f(x): return x + 1 " * 20
    train = tmp_path / "train.jsonl"
    benchmark = tmp_path / "bench.jsonl"
    write_jsonl(train, [{"id": "train-1", "text": shared}])
    write_jsonl(
        benchmark,
        [
            {"id": "exact", "fields": {"prompt": shared}},
            {"id": "near", "fields": {"prompt": shared + "changed"}},
            {"id": "clean", "fields": {"prompt": "unrelated story about a bridge"}},
        ],
    )

    documents, pieces, rows = index_training([train], 32)
    result = scan_benchmark(benchmark, documents, pieces, 32, 0.8)

    assert rows == 1
    assert result["exact_matches"] == ["exact"]
    assert [row["id"] for row in result["near_matches"]] == ["exact", "near"]


def test_normalization_and_short_shingles_are_stable() -> None:
    assert normalize_text(" A\n B\tC ") == "a b c"
    assert len(shingles("short", 32)) == 1

"""Build a train-only canonical token vocabulary for RPLAN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from floorflow.data.floorplan_token_targets import (
    build_floorplan_token_vocabulary,
    training_token_sequence,
)
from floorflow.data.rplan_boundary_adapter import (
    load_rplan_rows,
    rplan_row_to_canonical,
    rplan_taxonomy,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--numeric-bin-size", type=int, default=8)
    parser.add_argument(
        "--geometry-token-mode",
        choices=("reference", "quantized", "exact"),
        default="reference",
    )
    args = parser.parse_args()
    taxonomy = rplan_taxonomy()
    rows = load_rplan_rows(args.dataset, offset=args.offset, limit=args.limit)
    sequences = []
    for index, row in enumerate(rows):
        plan = rplan_row_to_canonical(row, taxonomy, image_size=256)
        sequences.append(
            training_token_sequence(
                plan,
                taxonomy,
                numeric_bin_size=args.numeric_bin_size,
                geometry_token_mode=args.geometry_token_mode,
            )
        )
        if (index + 1) % 1000 == 0:
            print(f"tokenized {index + 1}/{len(rows)}", flush=True)
    vocabulary, statistics = build_floorplan_token_vocabulary(sequences)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary_path = args.output_dir / "vocabulary.json"
    vocabulary.write(vocabulary_path)
    statistics_path = args.output_dir / "statistics.json"
    statistics_path.write_text(
        json.dumps(statistics.to_json(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "protocol": statistics.to_json(),
        "dataset": str(args.dataset),
        "dataset_sha256": _file_sha256(args.dataset),
        "offset": args.offset,
        "limit": args.limit,
        "numeric_bin_size": args.numeric_bin_size,
        "geometry_token_mode": args.geometry_token_mode,
        "vocabulary_size": len(vocabulary.tokens),
        "model_vocab_size": vocabulary.model_vocab_size,
        "vocabulary_sha256": vocabulary.sha256(),
        "vocabulary_path": str(vocabulary_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

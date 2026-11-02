#!/usr/bin/env python3
"""Fail closed if benchmark evaluation data overlaps the training corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.benchmark_contamination import index_training, scan_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / ".data/nlp/benchmarks_v1",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_benchmark_contamination_v1.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_paths = [ROOT / path for path in config["training_sources"]]
    missing = [str(path) for path in train_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"training sources are missing: {missing}")
    documents, pieces, training_rows = index_training(
        train_paths,
        int(config["shingle_characters"]),
    )
    manifest = json.loads((args.benchmark_root / "manifest.json").read_text(encoding="utf-8"))
    scans = {}
    failures = []
    for record in manifest["benchmarks"]:
        result = scan_benchmark(
            Path(record["path"]),
            documents,
            pieces,
            int(config["shingle_characters"]),
            float(config["near_overlap_threshold"]),
        )
        scans[record["id"]] = result
        if len(result["exact_matches"]) > int(config["exact_overlap_limit"]):
            failures.append(f"{record['id']} has exact contamination")
        if len(result["near_matches"]) > int(config["near_overlap_limit"]):
            failures.append(f"{record['id']} has near contamination")
    report = {
        "version": "nlp-benchmark-contamination-audit-v1",
        "status": "pass" if not failures else "fail",
        "errors": failures,
        "training_rows": training_rows,
        "training_documents": len(documents),
        "training_shingles": len(pieces),
        "benchmark_scans": scans,
        "evidence_boundary": config["evidence_boundary"],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

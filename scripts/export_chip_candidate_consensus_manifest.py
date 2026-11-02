#!/usr/bin/env python3
"""Merge saved chip case artifacts into a candidate-consensus manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.candidate_manifest import (
    candidate_manifest_from_sources,
)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("source must use NAME=PATH")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        type=_named_path,
        required=True,
    )
    parser.add_argument("--reference-source", default=None)
    parser.add_argument("--case-key", default=None)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = {}
    store = PackedCaseStore(args.cache) if args.cache is not None else None
    for name, path in args.source:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if args.case_key is not None:
            if not isinstance(payload, dict) or args.case_key not in payload:
                raise ValueError(
                    f"source {name} lacks case key {args.case_key}"
                )
            rows = payload[args.case_key]
        else:
            rows = payload.get("cases") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"source {name} contains no case rows")
        if store is not None:
            for row in rows:
                if "constraints" in row:
                    continue
                batch = store.batch(
                    np.asarray([int(row["index"])], dtype=np.int64),
                    device="cpu",
                )
                count = int(row["blocks"])
                row["constraints"] = (
                    batch["constraints"][0, :count].numpy().tolist()
                )
        source_rows[name] = rows
    cases = candidate_manifest_from_sources(
        source_rows,
        reference_source=args.reference_source,
    )
    report = {
        "version": "chip-candidate-consensus-manifest-v1",
        "sources": {name: str(path) for name, path in args.source},
        "reference_source": args.reference_source,
        "case_key": args.case_key,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()

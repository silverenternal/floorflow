#!/usr/bin/env python3
"""Build candidate-consensus NPZ shards from a frozen candidate manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.chip_flow.candidate_consensus_training import (
    write_candidate_consensus_cohort,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("candidate manifest requires a non-empty cases list")
    paths = write_candidate_consensus_cohort(args.output_dir, cases)
    report = {
        "status": "complete",
        "manifest": str(args.manifest),
        "cases": len(cases),
        "shards": [str(path) for path in paths],
        "contract": "one fixed shape case per NPZ shard",
    }
    (args.output_dir / "manifest.report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

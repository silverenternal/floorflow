#!/usr/bin/env python3
"""Build an auditable room failure taxonomy from HI-00 rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from floorflow.evaluation.failure_taxonomy import summarise_failure_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragmentation-threshold", type=float, default=None)
    args = parser.parse_args()
    payload = json.loads(args.decomposition.read_text(encoding="utf-8"))
    rows = payload["cases"]
    threshold = args.fragmentation_threshold
    if threshold is None:
        threshold = float(
            np.quantile([row["support_fragmentation_gap"] for row in rows], 0.75)
        )
    report = summarise_failure_taxonomy(
        rows,
        fragmentation_threshold=threshold,
    )
    report["fragmentation_threshold"] = threshold
    report["source"] = str(args.decomposition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                name: values["fraction"]
                for name, values in report["categories"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

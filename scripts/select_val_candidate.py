#!/usr/bin/env python3
"""Select a candidate on validation metrics or fail closed to reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.training.val_selection import evaluate_candidate


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_named_path,
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = json.loads(args.reference_metrics.read_text(encoding="utf-8"))
    rows = []
    for name, path in args.candidate:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        decision = evaluate_candidate(reference, candidate)
        rows.append(
            {
                "name": name,
                "metrics": str(path),
                "summary": candidate["metrics"],
                **decision.as_dict(),
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: float(row["summary"]["room_macro_iou"]),
        default={"name": args.reference_name, "metrics": str(args.reference_metrics)},
    )
    report = {
        "selection_split": "val",
        "selection_metric": "fixed guards plus overlay improvement",
        "best_of_n_used": False,
        "reference": {
            "name": args.reference_name,
            "metrics": str(args.reference_metrics),
        },
        "candidates": rows,
        "selected": selected,
        "rolled_back": selected["name"] == args.reference_name,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

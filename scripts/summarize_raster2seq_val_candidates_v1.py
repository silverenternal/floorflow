"""Summarize validation-only Raster2Seq identity candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("room_f1", "room_iou", "corner_f1", "room_sem_f1", "window_door_f1")


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must use name=path")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=parse_candidate,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster2seq_val_selection_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate_paths = dict(args.candidate)
    candidates = {
        name: json.loads(candidate_paths[name].read_text(encoding="utf-8"))
        for name, _path in args.candidate
    }
    rows = {}
    for name, values in candidates.items():
        rows[name] = {
            "source": str(candidate_paths[name]),
            "metrics": {metric: float(values[metric]) for metric in METRICS},
            "delta": {metric: float(values[metric]) - float(baseline[metric]) for metric in METRICS},
            "passes_non_degradation": (
                float(values["room_iou"]) >= float(baseline["room_iou"])
                and float(values["corner_f1"]) >= float(baseline["corner_f1"])
            ),
        }
    eligible = {
        name: row
        for name, row in rows.items()
        if row["metrics"]["room_f1"] > float(baseline["room_f1"])
        and row["passes_non_degradation"]
    }
    selected = max(
        eligible,
        key=lambda name: (
            rows[name]["metrics"]["room_f1"],
            rows[name]["metrics"]["room_iou"],
        ),
        default="baseline",
    )
    report = {
        "status": "complete",
        "selection_split": "val",
        "official_test509_used": False,
        "best_of_n_used": False,
        "policy": (
            "Select the highest val RoomF1 candidate that also preserves or "
            "improves RoomIoU and corner F1 over the validation anchor."
        ),
        "baseline": {metric: float(baseline[metric]) for metric in METRICS},
        "candidates": rows,
        "eligible": sorted(eligible),
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "eligible": sorted(eligible)}, indent=2))


if __name__ == "__main__":
    main()

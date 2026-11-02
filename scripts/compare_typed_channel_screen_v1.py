"""Compare matched typed-channel screens on case-level validation rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = {
    "room_macro_iou": False,
    "instance_f1": False,
    "instance_type_accuracy": False,
    "instance_count_abs_error": True,
    "wall_f1": False,
    "opening_f1": False,
    "furniture_f1": False,
    "node_f1": False,
    "edge_overlap": False,
    "ged": True,
}


def _load_selected_rows(run: Path) -> tuple[dict, dict[str, dict]]:
    report_path = run / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        history = [
            json.loads(line)
            for line in (run / "train.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        selected = max(history, key=lambda row: row["selection_score"])
        report = {
            "selected_epoch": int(selected["epoch"]),
            "history_complete": False,
        }
    checkpoint = f"epoch_{int(report['selected_epoch']):02d}"
    rows = {}
    with (run / "atom_proxy_rows.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("checkpoint") != checkpoint:
                continue
            rows[str(row["case_id"])] = row
    if not rows:
        raise ValueError(f"no rows found for {checkpoint} under {run}")
    return report, rows


def _load_rows(path: Path) -> tuple[dict, dict[str, dict]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["case_id"])] = row
    if not rows:
        raise ValueError(f"no rows found under {path}")
    checkpoint = str(next(iter(rows.values())).get("checkpoint", "rowset"))
    return {"selected_epoch": checkpoint}, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--candidate-rows", type=Path)
    parser.add_argument("--control-rows", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()

    if (args.candidate is None) == (args.candidate_rows is None):
        parser.error("provide exactly one candidate run or candidate row file")
    if (args.control is None) == (args.control_rows is None):
        parser.error("provide exactly one control run or control row file")
    candidate_report, candidate_rows = (
        _load_rows(args.candidate_rows)
        if args.candidate_rows is not None
        else _load_selected_rows(args.candidate)
    )
    control_report, control_rows = (
        _load_rows(args.control_rows)
        if args.control_rows is not None
        else _load_selected_rows(args.control)
    )
    case_ids = sorted(candidate_rows)
    if case_ids != sorted(control_rows):
        raise ValueError("candidate and control must contain identical cases")
    current = [
        {"index": index, **candidate_rows[case]["canonical_metrics"]}
        for index, case in enumerate(case_ids)
    ]
    baseline = [
        {"index": index, **control_rows[case]["canonical_metrics"]}
        for index, case in enumerate(case_ids)
    ]
    comparisons = {
        name: paired_delta(
            current,
            baseline,
            name,
            iterations=args.iterations,
            seed=args.seed,
            lower_is_better=lower_is_better,
        )
        for name, lower_is_better in METRICS.items()
    }
    report = {
        "schema_version": "typed-channel-matched-screen-v1",
        "candidate": str(args.candidate_rows or args.candidate),
        "control": str(args.control_rows or args.control),
        "candidate_selected_epoch": candidate_report["selected_epoch"],
        "control_selected_epoch": control_report["selected_epoch"],
        "cases": len(case_ids),
        "metrics": comparisons,
        "direction_gates": {
            name: (
                float(result["ci_high"]) < 0.0
                if METRICS[name]
                else float(result["ci_low"]) > 0.0
            )
            for name, result in comparisons.items()
        },
        "claim_boundary": (
            "Single-seed mechanism screen only. Safety and disjoint-holdout "
            "gates remain required before promotion."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

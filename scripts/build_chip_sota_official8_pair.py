#!/usr/bin/env python3
"""Compare candidate and Trinity on the predeclared eight Official-100 cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_sota_official8_candidate_nfe4_refine3200.json",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_sota_official8_trinity_nfe4_refine3200.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/reports/chip_sota_official8_manifest_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_sota_official8_pair_v1.json",
    )
    parser.add_argument("--candidate-key", default="nfe4_refine3200")
    parser.add_argument("--refiner-steps", type=int, default=3200)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_metric(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    args = parse_args()
    candidate = load(args.candidate)
    anchor = load(args.anchor)
    manifest = load(args.manifest)
    candidate_rows = load(
        args.candidate.with_name(f"{args.candidate.stem}_cases.json")
    )[args.candidate_key]
    anchor_rows = anchor["rows"]
    expected = set(manifest["indices"])
    candidate_by_index = {int(row["index"]): row for row in candidate_rows}
    anchor_by_index = {int(row["index"]): row for row in anchor_rows}
    if not expected.issubset(candidate_by_index):
        raise ValueError("candidate rows do not cover the frozen manifest")
    if not expected.issubset(anchor_by_index):
        raise ValueError("anchor rows do not cover the frozen manifest")
    candidate_rows = [candidate_by_index[index] for index in sorted(expected)]
    anchor_rows = [anchor_by_index[index] for index in sorted(expected)]

    metrics = ("legalized_hard_cost", "legalized_v_rel")
    candidate_mean = {key: mean_metric(candidate_rows, key) for key in metrics}
    anchor_mean = {key: mean_metric(anchor_rows, key) for key in metrics}
    paired = {
        key: paired_delta(
            candidate_rows,
            anchor_rows,
            key,
            iterations=10000,
            seed=20261031 + index,
            lower_is_better=True,
        )
        for index, key in enumerate(metrics)
    }
    candidate_feasible = float(
        candidate["results"][args.candidate_key]["legalized_feasible_fraction"]
    )
    anchor_feasible = float(anchor["feasible_fraction"])
    point_win = all(candidate_mean[key] < anchor_mean[key] for key in metrics)
    ci_win = all(paired[key]["ci_high"] < 0.0 for key in metrics)
    report = {
        "version": "chip-sota-official8-pair-v1",
        "status": (
            "pilot_candidate_win_ci_excludes_zero"
            if point_win and ci_win and candidate_feasible == anchor_feasible == 1.0
            else "pilot_candidate_point_win_ci_pending"
            if point_win and candidate_feasible == anchor_feasible == 1.0
            else "pilot_not_supported"
        ),
        "scope": "8 predeclared Official-100 cases spanning block counts 21-120",
        "budget": {
            "candidate_parameters": 5836447,
            "anchor_parameters": 5856400,
            "nfe": 4,
            "refiner_steps": args.refiner_steps,
            "samples": 1,
        },
        "candidate": {
            "path": str(args.candidate),
            "mean": candidate_mean,
            "feasible_fraction": candidate_feasible,
        },
        "anchor": {
            "path": str(args.anchor),
            "mean": anchor_mean,
            "feasible_fraction": anchor_feasible,
        },
        "relative_delta": {
            key: candidate_mean[key] / anchor_mean[key] - 1.0
            for key in metrics
        },
        "paired": paired,
        "claim_boundary": (
            "Eight-case development pilot only. A point or CI win cannot be "
            "promoted to SOTA until confirmed on the full frozen Official-100 "
            "and independent external designs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate": candidate_mean,
                "anchor": anchor_mean,
                "relative_delta": report["relative_delta"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

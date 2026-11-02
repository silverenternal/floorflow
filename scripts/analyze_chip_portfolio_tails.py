#!/usr/bin/env python3
"""Decompose a chip portfolio against its matched Trinity anchor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "cost",
    "hpwl_gap",
    "area_gap",
    "v_rel",
    "b2b_hpwl_gap",
    "p2b_hpwl_gap",
    "grouping",
    "boundary",
    "rho_mae",
)
QUANTILES = (0.50, 0.75, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trinity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="chip-portfolio-tail-analysis")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metric(row: dict, key: str) -> float:
    value = row.get(f"legalized_{key}")
    return float(row[key] if value is None else value)


def candidate_rows(report: dict) -> list[dict]:
    rows = report.get("selected_rows", report.get("rows", []))
    return sorted(rows, key=lambda row: int(row["index"]))


def trinity_rows(report: dict) -> list[dict]:
    return sorted(report["rows"], key=lambda row: int(row["index"]))


def relative(current: float, reference: float) -> float | None:
    if abs(reference) <= 1.0e-12:
        return None
    return current / reference - 1.0


def mean_components(candidate: list[dict], reference: list[dict]) -> dict:
    result = {}
    for key in METRICS:
        candidate_mean = float(np.mean([metric(row, key) for row in candidate]))
        reference_mean = float(np.mean([metric(row, key) for row in reference]))
        result[key] = {
            "candidate": candidate_mean,
            "trinity": reference_mean,
            "delta": candidate_mean - reference_mean,
            "relative": relative(candidate_mean, reference_mean),
        }
    return result


def paired_components(
    candidate: list[dict],
    reference: list[dict],
    iterations: int,
) -> dict:
    normalized_candidate = [
        {"index": int(row["index"]), **{key: metric(row, key) for key in METRICS}}
        for row in candidate
    ]
    normalized_reference = [
        {"index": int(row["index"]), **{key: metric(row, key) for key in METRICS}}
        for row in reference
    ]
    return {
        key: paired_delta(
            normalized_candidate,
            normalized_reference,
            key,
            iterations=iterations,
            seed=7400 + offset,
            lower_is_better=True,
        )
        for offset, key in enumerate(METRICS)
    }


def quantiles(candidate: list[dict], reference: list[dict]) -> dict:
    result = {}
    for key in ("cost", "v_rel", "area_gap", "b2b_hpwl_gap"):
        candidate_values = np.asarray([metric(row, key) for row in candidate])
        reference_values = np.asarray([metric(row, key) for row in reference])
        candidate_quantiles = np.quantile(candidate_values, QUANTILES)
        reference_quantiles = np.quantile(reference_values, QUANTILES)
        result[key] = {
            str(quantile): {
                "candidate": float(current),
                "trinity": float(baseline),
                "relative": relative(float(current), float(baseline)),
            }
            for quantile, current, baseline in zip(
                QUANTILES,
                candidate_quantiles,
                reference_quantiles,
                strict=True,
            )
        }
    return result


def strata(candidate: list[dict], reference: list[dict]) -> dict:
    result = {}
    low = min(int(row["blocks"]) for row in reference)
    high = max(int(row["blocks"]) for row in reference)
    for start in range(low, high + 1, 20):
        end = min(start + 19, high)
        candidate_subset = [
            current
            for current, baseline in zip(candidate, reference, strict=True)
            if start <= int(baseline["blocks"]) <= end
        ]
        reference_subset = [
            row for row in reference if start <= int(row["blocks"]) <= end
        ]
        result[f"{start}-{end}"] = {
            "cases": len(candidate_subset),
            "relative": {
                key: relative(
                    float(np.mean([metric(row, key) for row in candidate_subset])),
                    float(np.mean([metric(row, key) for row in reference_subset])),
                )
                for key in ("cost", "v_rel", "area_gap", "b2b_hpwl_gap")
            },
            "delta": {
                key: (
                    float(np.mean([metric(row, key) for row in candidate_subset]))
                    - float(np.mean([metric(row, key) for row in reference_subset]))
                )
                for key in ("cost", "v_rel", "area_gap", "b2b_hpwl_gap")
            },
        }
    return result


def worst_cases(candidate: list[dict], reference: list[dict], limit: int) -> list[dict]:
    rows = []
    for current, baseline in zip(candidate, reference, strict=True):
        loss = metric(current, "cost") - metric(baseline, "cost")
        rows.append(
            {
                "index": int(current["index"]),
                "blocks": int(baseline["blocks"]),
                "selection": current.get("selection"),
                "cost_delta": loss,
                "cost": metric(current, "cost"),
                "trinity_cost": metric(baseline, "cost"),
                "area_gap_delta": (
                    metric(current, "area_gap") - metric(baseline, "area_gap")
                ),
                "b2b_hpwl_gap_delta": (
                    metric(current, "b2b_hpwl_gap")
                    - metric(baseline, "b2b_hpwl_gap")
                ),
                "v_rel_delta": metric(current, "v_rel") - metric(baseline, "v_rel"),
            }
        )
    return sorted(rows, key=lambda row: row["cost_delta"], reverse=True)[:limit]


def main() -> None:
    args = parse_args()
    candidate = candidate_rows(load(args.candidate))
    reference = trinity_rows(load(args.trinity))
    if len(candidate) != len(reference):
        raise ValueError("candidate and Trinity reports must cover the same cases")
    if [int(row["index"]) for row in candidate] != [
        int(row["index"]) for row in reference
    ]:
        raise ValueError("candidate and Trinity case indices differ")
    report = {
        "version": "chip-portfolio-tail-analysis-v1",
        "label": args.label,
        "candidate_report": str(args.candidate),
        "trinity_report": str(args.trinity),
        "cases": len(candidate),
        "components": mean_components(candidate, reference),
        "paired": paired_components(candidate, reference, args.bootstrap),
        "quantiles": quantiles(candidate, reference),
        "strata": strata(candidate, reference),
        "worst_cost_cases": worst_cases(candidate, reference, args.top_k),
        "claim_boundary": (
            "This diagnostic decomposes an already-selected portfolio. "
            "Selection provenance is reported when available; do not use "
            "Trinity outcomes to reroute individual cases."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": report["cases"],
                "components": {
                    key: value["relative"] for key, value in report["components"].items()
                },
                "strata": report["strata"],
                "worst_case": report["worst_cost_cases"][0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the paired 5k Trinity token-flow screen report."""

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
        / "outputs/chip_trinity/screen_5k/trinity_token_seed19_5k",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT
        / "outputs/chip_graph_mechanism/screen_5k/graph_full_seed19_5k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_trinity_token_flow_screen_v1.json",
    )
    parser.add_argument("--gate", choices=("screen", "rescue"), default="screen")
    parser.add_argument("--min-refined-improvement", type=float, default=None)
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def endpoint_report(arm: Path, kind: str) -> dict:
    if kind == "raw":
        return read(arm / "official_test100_nfe8_legalized.json")
    return read(arm / "refiner_nfe4_ref400.json")["results"]["nfe4_refine400"]


def endpoint_rows(arm: Path, kind: str) -> list[dict]:
    if kind == "raw":
        return read(arm / "official_test100_nfe8_legalized_cases.json")
    return read(arm / "refiner_nfe4_ref400_cases.json")["nfe4_refine400"]


def endpoint_metrics(report: dict) -> dict[str, float]:
    return {
        "legalized_hard_cost": float(report["legalized_hard_cost"]),
        "legalized_v_rel": float(report["legalized_v_rel"]),
        "legalized_feasible_fraction": float(
            report["legalized_feasible_fraction"]
        ),
    }


def main() -> None:
    args = parse_args()
    minimum = (
        args.min_refined_improvement
        if args.min_refined_improvement is not None
        else (0.05 if args.gate == "screen" else 0.0)
    )
    baseline = {
        kind: endpoint_report(args.baseline, kind)
        for kind in ("raw", "refined")
    }
    candidate = {
        kind: endpoint_report(args.candidate, kind)
        for kind in ("raw", "refined")
    }
    comparisons = {}
    for kind in ("raw", "refined"):
        baseline_metrics = endpoint_metrics(baseline[kind])
        candidate_metrics = endpoint_metrics(candidate[kind])
        paired = {}
        for index, metric in enumerate(
            ("legalized_hard_cost", "legalized_v_rel")
        ):
            paired[metric] = paired_delta(
                endpoint_rows(args.candidate, kind),
                endpoint_rows(args.baseline, kind),
                metric,
                iterations=args.bootstrap,
                seed=20261101 + index + (0 if kind == "raw" else 10),
                lower_is_better=True,
            )
        comparisons[kind] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "relative_delta": {
                metric: candidate_metrics[metric] / baseline_metrics[metric] - 1.0
                for metric in ("legalized_hard_cost", "legalized_v_rel")
            },
            "paired": paired,
        }
    checks = {
        "refined_cost_meets_gate": (
            comparisons["refined"]["relative_delta"]["legalized_hard_cost"]
            <= -minimum
        ),
        "refined_v_rel_meets_gate": (
            comparisons["refined"]["relative_delta"]["legalized_v_rel"]
            <= -minimum
        ),
    }
    report = {
        "status": (
            "pass"
            if all(checks.values())
            else f"reject_{args.gate}_token_flow"
        ),
        "scope": f"seed-19 {args.gate} matched data/objective/runtime comparison",
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "minimum_refined_improvement": minimum,
        "comparisons": comparisons,
        "checks": checks,
        "decision": (
            (
                "promote to the conditional 20k gate"
                if args.gate == "screen"
                else "proceed to three-seed promotion"
            )
            if all(checks.values())
            else "reject promotion and retain the negative result"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

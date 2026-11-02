#!/usr/bin/env python3
"""Compare graph features and topology at one training budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "graph_full_seed19_5k",
    "graph_feature_only_seed19_5k",
    "graph_topology_only_seed19_5k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs/chip_graph_mechanism/screen_5k",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_graph_mechanism_ablation_v1.json",
    )
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def endpoint_rows(arm_dir: Path, kind: str) -> list[dict]:
    if kind == "raw":
        return read(arm_dir / "official_test100_nfe8_legalized_cases.json")
    return read(arm_dir / "refiner_nfe4_ref400_cases.json")["nfe4_refine400"]


def endpoint_summary(arm_dir: Path, kind: str) -> dict:
    if kind == "raw":
        report = read(arm_dir / "official_test100_nfe8_legalized.json")
    else:
        report = read(arm_dir / "refiner_nfe4_ref400.json")["results"][
            "nfe4_refine400"
        ]
    return {
        "legalized_hard_cost": float(report["legalized_hard_cost"]),
        "legalized_v_rel": float(report["legalized_v_rel"]),
        "legalized_feasible_fraction": float(
            report["legalized_feasible_fraction"]
        ),
    }


def paired_metrics(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    iterations: int,
) -> dict:
    output = {}
    for index, metric in enumerate(
        ("legalized_hard_cost", "legalized_v_rel")
    ):
        output[metric] = paired_delta(
            candidate_rows,
            baseline_rows,
            metric,
            iterations=iterations,
            seed=20261031 + index,
            lower_is_better=True,
        )
    return output


def main() -> None:
    args = parse_args()
    baseline = ARMS[0]
    baseline_dir = args.root / baseline
    arms = {}
    for arm in ARMS:
        arm_dir = args.root / arm
        training = read(args.root / f"{arm}.json")
        arms[arm] = {
            "parameters": int(training["model_parameters"]),
            "dev_total": float(training["final_dev"]["total"]),
            "raw": endpoint_summary(arm_dir, "raw"),
            "refined": endpoint_summary(arm_dir, "refined"),
        }
        if arm != baseline:
            arms[arm]["paired_vs_full"] = {
                kind: paired_metrics(
                    endpoint_rows(baseline_dir, kind),
                    endpoint_rows(arm_dir, kind),
                    iterations=args.bootstrap,
                )
                for kind in ("raw", "refined")
            }

    strongest = None
    for arm in ARMS[1:]:
        score = (
            arms[arm]["refined"]["legalized_hard_cost"]
            + arms[arm]["refined"]["legalized_v_rel"]
        )
        if strongest is None or score < strongest[0]:
            strongest = (score, arm)
    assert strongest is not None
    strongest_arm = strongest[1]
    checks = {
        "full_beats_strongest_single_mechanism_cost": (
            arms[baseline]["refined"]["legalized_hard_cost"]
            < arms[strongest_arm]["refined"]["legalized_hard_cost"]
        ),
        "full_beats_strongest_single_mechanism_v_rel": (
            arms[baseline]["refined"]["legalized_v_rel"]
            < arms[strongest_arm]["refined"]["legalized_v_rel"]
        ),
        "all_full_paired_intervals_prefer_full": all(
            arms[arm]["paired_vs_full"]["refined"][metric]["ci_high"] > 0.0
            for arm in ARMS[1:]
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "narrow_claim",
        "scope": "seed-19 5k mechanism attribution; matched data/objective/runtime",
        "root": str(args.root),
        "baseline": baseline,
        "strongest_single_mechanism": strongest_arm,
        "arms": arms,
        "checks": checks,
        "decision": (
            "full graph-conditioned mechanism is supported"
            if all(checks.values())
            else "attribute the gain to the strongest observed feature or topology arm"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

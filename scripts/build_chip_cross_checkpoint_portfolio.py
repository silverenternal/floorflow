#!/usr/bin/env python3
"""Evaluate target-free selection over existing cross-checkpoint candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.legalizer_portfolio import (
    LegalizerCandidate,
    select_legalizer_candidate,
)
from floorflow.evaluation.paired_bootstrap import paired_delta

REPORTS = {
    "M48": ROOT / "outputs/reports/chip_m48_portfolio_dev32_postfix_v1.json",
    "M61B": ROOT / "outputs/reports/chip_m72_m61b_portfolio_dev32_v1.json",
    "M65": ROOT / "outputs/reports/chip_m72_m65_portfolio_dev32_v1.json",
}
VARIANTS = ("baseline", "expanded", "projector32")
METRICS = (
    "cost",
    "v_rel",
    "area_gap",
    "b2b_hpwl_gap",
    "p2b_hpwl_gap",
    "grouping",
    "boundary",
    "rho_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m72_cross_checkpoint_portfolio_v1.json",
    )
    parser.add_argument("--m48", type=Path, default=REPORTS["M48"])
    parser.add_argument("--m61b", type=Path, default=REPORTS["M61B"])
    parser.add_argument("--m65", type=Path, default=REPORTS["M65"])
    parser.add_argument(
        "--policy",
        choices=(
            "legality_first",
            "balanced",
            "geometry_consensus",
            "legality_guarded_consensus",
        ),
        default="legality_first",
    )
    parser.add_argument("--vrel-tolerance", type=float, default=0.01)
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def view(name: str, entry: dict) -> LegalizerCandidate:
    legalized = entry["stages"]["legalized"]
    geometry = entry.get("target_free_geometry", {})
    return LegalizerCandidate(
        name=name,
        v_rel=float(legalized["v_rel"]),
        displacement=float(entry["displacement"]["center_l2"]),
        boundary=float(legalized["boundary"]),
        grouping=float(legalized["grouping"]),
        rung=int(entry["legalizer"]["rung"]),
        solves=int(entry["legalizer"]["solves"]),
        b2b_hpwl=(
            float(geometry["b2b_hpwl"]) if "b2b_hpwl" in geometry else None
        ),
        p2b_hpwl=(
            float(geometry["p2b_hpwl"]) if "p2b_hpwl" in geometry else None
        ),
        outline_compactness=(
            float(geometry["outline_compactness"])
            if "outline_compactness" in geometry
            else None
        ),
        outline_aspect=(
            float(geometry["outline_aspect"])
            if "outline_aspect" in geometry
            else None
        ),
    )


def legalized(entry: dict) -> dict:
    return entry["stages"]["legalized"]


def paired(
    current: list[dict],
    reference: list[dict],
    iterations: int,
) -> dict:
    return {
        metric: paired_delta(
            current,
            reference,
            metric,
            iterations=iterations,
            seed=2300 + offset,
            lower_is_better=True,
        )
        for offset, metric in enumerate(METRICS)
    }


def main() -> None:
    args = parse_args()
    report_paths = {
        "M48": args.m48,
        "M61B": args.m61b,
        "M65": args.m65,
    }
    reports = {name: load(path) for name, path in report_paths.items()}
    rows = {
        name: {int(row["index"]): row for row in report["rows"]}
        for name, report in reports.items()
    }
    indices = sorted(set.intersection(*(set(value) for value in rows.values())))
    selected_rows = []
    reference_rows = []
    selection_counts = {f"{name}:{variant}": 0 for name in REPORTS for variant in VARIANTS}
    for index in indices:
        reference_entry = rows["M48"][index]["baseline"]
        baseline = view("M48:baseline", reference_entry)
        candidates = tuple(
            view(f"{name}:{variant}", rows[name][index][variant])
            for name in REPORTS
            for variant in VARIANTS
            if not (name == "M48" and variant == "baseline")
        )
        selected = select_legalizer_candidate(
            baseline,
            candidates,
            policy=args.policy,
            vrel_tolerance=args.vrel_tolerance,
        )
        name, variant = selected.name.split(":", 1)
        entry = rows[name][index][variant]
        selection_counts[selected.name] += 1
        selected_metrics = {
            "index": index,
            "selection": selected.name,
            **legalized(entry),
        }
        selected_rows.append(selected_metrics)
        reference_rows.append(
            {"index": index, **legalized(rows["M48"][index]["selected"])}
        )
    relative = {
        metric: {
            "reference": sum(float(row[metric]) for row in reference_rows) / len(reference_rows),
            "candidate": sum(float(row[metric]) for row in selected_rows) / len(selected_rows),
        }
        for metric in METRICS
    }
    for metric in METRICS:
        relative[metric]["relative"] = (
            relative[metric]["candidate"] / relative[metric]["reference"] - 1.0
        )
    paired_metrics = paired(selected_rows, reference_rows, args.bootstrap)
    joint_primary = (
        relative["cost"]["relative"] < 0.0
        and relative["v_rel"]["relative"] < 0.0
    )
    structural = any(
        relative[metric]["relative"] < 0.0
        for metric in ("area_gap", "b2b_hpwl_gap", "boundary")
    )
    report = {
        "version": "chip-m72-cross-checkpoint-portfolio-v1",
        "status": "pass" if joint_primary and structural else "fail",
        "reports": {name: str(path) for name, path in report_paths.items()},
        "cases": len(indices),
        "selection_counts": {
            name: count for name, count in selection_counts.items() if count
        },
        "relative_to_m48_portfolio": relative,
        "paired_to_m48_portfolio": paired_metrics,
        "selected_rows": selected_rows,
        "joint_primary_improves": joint_primary,
        "structural_target_improves": structural,
        "budget_boundary": (
            "Uses all already-generated candidates from three checkpoints: "
            "six trajectories and nine legalizer calls. This is an upper-budget "
            "complementarity diagnostic, not a deployable low-cost setting."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "joint_primary": joint_primary,
                "selection_counts": report["selection_counts"],
                "relative": {
                    key: value["relative"] for key, value in relative.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

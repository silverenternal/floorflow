#!/usr/bin/env python3
"""Select target-free candidates across existing chip inference seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.legalizer_portfolio import select_legalizer_candidate
from floorflow.evaluation.paired_bootstrap import paired_delta
from scripts.build_chip_cross_checkpoint_portfolio import (
    METRICS,
    legalized,
    view,
)

VARIANTS = ("baseline", "expanded", "projector32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m48",
        type=Path,
        nargs="+",
        default=[
            ROOT / "outputs/reports/chip_m48_portfolio_official100_postfix_seed19.json",
            ROOT / "outputs/reports/chip_m74_m48_portfolio_seed20_official100_v1.json",
            ROOT / "outputs/reports/chip_m74_m48_portfolio_seed21_official100_v1.json",
        ],
    )
    parser.add_argument(
        "--m61b",
        type=Path,
        default=ROOT / "outputs/reports/chip_m73_m61b_portfolio_official100_v1.json",
    )
    parser.add_argument(
        "--m65",
        type=Path,
        default=ROOT / "outputs/reports/chip_m73_m65_portfolio_official100_v1.json",
    )
    parser.add_argument(
        "--fixed-reference",
        type=Path,
        default=ROOT / "outputs/reports/chip_m73_cross_checkpoint_portfolio_official100_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=100)
    parser.add_argument(
        "--policy",
        choices=(
            "legality_first",
            "balanced",
            "geometry_consensus",
            "legality_guarded_consensus",
        ),
        default="legality_guarded_consensus",
    )
    parser.add_argument("--vrel-tolerance", type=float, default=0.01)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def source_rows(path: Path) -> dict[int, dict]:
    return {int(row["index"]): row for row in load(path)["rows"]}


def paired(
    current: list[dict],
    reference: list[dict],
    metric: str,
    iterations: int,
) -> dict:
    return paired_delta(
        current,
        reference,
        metric,
        iterations=iterations,
        seed=7500 + METRICS.index(metric),
        lower_is_better=True,
    )


def main() -> None:
    args = parse_args()
    sources = {
        f"M48-s{seed}": source_rows(path)
        for seed, path in zip((19, 20, 21), args.m48, strict=True)
    }
    sources["M61B"] = source_rows(args.m61b)
    sources["M65"] = source_rows(args.m65)
    indices = sorted(set.intersection(*(set(rows) for rows in sources.values())))
    fixed_payload = load(args.fixed_reference)
    if "selected_rows" in fixed_payload:
        fixed = {
            int(row["index"]): row for row in fixed_payload["selected_rows"]
        }
    else:
        fixed = {
            int(row["index"]): {
                "index": int(row["index"]),
                **legalized(row["selected"]),
            }
            for row in fixed_payload["rows"]
        }
    selected_rows = []
    reference_rows = []
    selection_counts = {
        f"{source}:{variant}": 0
        for source in sources
        for variant in VARIANTS
    }
    missing_geometry = 0
    solver_escalations = []
    for index in indices:
        baseline = view("M48-s19:baseline", sources["M48-s19"][index]["baseline"])
        candidate_entries = []
        for source, rows in sources.items():
            for variant in VARIANTS:
                if source == "M48-s19" and variant == "baseline":
                    continue
                entry = rows[index][variant]
                candidate_entries.append((source, variant, entry))
        candidates = tuple(
            view(f"{source}:{variant}", entry)
            for source, variant, entry in candidate_entries
        )
        selected = select_legalizer_candidate(
            baseline,
            candidates,
            policy=args.policy,
            vrel_tolerance=args.vrel_tolerance,
        )
        source, variant = selected.name.split(":", 1)
        entry = sources[source][index][variant]
        geometry = entry.get("target_free_geometry", {})
        if not geometry:
            missing_geometry += 1
        if (
            int(entry["legalizer"]["rung"]) > baseline.rung
            or int(entry["legalizer"]["solves"]) > baseline.solves
        ):
            solver_escalations.append(index)
        selection_counts[selected.name] += 1
        selected_rows.append(
            {
                "index": index,
                "selection": selected.name,
                **legalized(entry),
            }
        )
        reference_rows.append(fixed[index])
    if len(selected_rows) != args.expected_cases:
        raise ValueError(
            "seed-diverse selection case count mismatch: "
            f"{len(selected_rows)} != {args.expected_cases}"
        )
    deltas = {
        metric: paired(selected_rows, reference_rows, metric, args.bootstrap)
        for metric in METRICS
    }
    report = {
        "version": "chip-m75-seed-diverse-portfolio-v1",
        "status": "complete",
        "policy": args.policy,
        "vrel_tolerance": args.vrel_tolerance,
        "cases": len(selected_rows),
        "sources": {
            "m48": [str(path) for path in args.m48],
            "m61b": str(args.m61b),
            "m65": str(args.m65),
        },
        "selection_counts": {
            name: count for name, count in selection_counts.items() if count
        },
        "selected_with_missing_target_free_geometry": missing_geometry,
        "solver_escalations": solver_escalations,
        "selected_feasible_fraction": sum(
            bool(row["feasible"]) for row in selected_rows
        )
        / len(selected_rows),
        "paired_to_fixed_seed_reference": deltas,
        "selected_rows": selected_rows,
        "selection_contract": {
            "uses_target": False,
            "uses_trinity_outcomes": False,
            "within_case_only": True,
            "budget_reference": "M48 seed19 baseline rung/solves",
            "features": [
                "v_rel",
                "b2b_hpwl",
                "p2b_hpwl",
                "outline_compactness",
                "outline_aspect",
            ],
        },
        "budget": {
            "m48_reports": 3,
            "trajectories": 10,
            "nfe_total": 14,
            "refiner_steps_total": 16000,
            "legalizer_calls": 15,
        },
        "boundary": (
            "Multi-seed inference-time candidate search. Not same-inference "
            "and not a training-seed replication."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": report["cases"],
                "selection_counts": report["selection_counts"],
                "missing_geometry": missing_geometry,
                "solver_escalations": solver_escalations,
                "selected_feasible_fraction": report["selected_feasible_fraction"],
                "relative_to_fixed": {
                    metric: value["relative_delta"]
                    for metric, value in deltas.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

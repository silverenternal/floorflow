#!/usr/bin/env python3
"""Stratify chip layout failures by scale and native FloorSet constraints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from floorflow.chip_flow.cache import PackedCaseStore

ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "flat_dense": "outputs/chip_large/v2_20k/dense_no_ot_huber/official_test100_nfe8_legalized_cases.json",
    "global_slot": "outputs/chip_slot/global_v1/global_slot_overlap10_20k/official_test100_nfe8_legalized_cases.json",
    "fixed_hierarchy": "outputs/chip_hierarchical/v2_matched/hierarchical_slot_overlap10_20k/official_test100_nfe8_legalized_cases.json",
}

METRICS = (
    "raw_hpwl_gap",
    "raw_area_gap",
    "raw_v_rel",
    "latent_mse",
    "legalized_hard_cost",
    "legalized_hpwl_gap",
    "legalized_area_gap",
    "legalized_v_rel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_constraint_failure_taxonomy_v1.json",
    )
    return parser.parse_args()


def read_cases(path: Path) -> dict[int, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["index"]): row for row in rows}


def case_flags(store: PackedCaseStore, index: int, blocks: int) -> dict:
    constraints = store.arrays["constraints"][index, :blocks]
    _, mib_counts = np.unique(
        constraints[constraints[:, 2] > 0, 2],
        return_counts=True,
    )
    _, cluster_counts = np.unique(
        constraints[constraints[:, 3] > 0, 3],
        return_counts=True,
    )
    flags = {
        "blocks": int(blocks),
        "length": "21-60" if blocks <= 60 else "61-120",
        "fixed": bool(np.any(constraints[:, 0] != 0)),
        "preplaced": bool(np.any(constraints[:, 1] != 0)),
        "mib": bool(np.any(mib_counts > 1)),
        "cluster": bool(np.any(cluster_counts > 1)),
        "boundary": bool(np.any(constraints[:, 4] != 0)),
    }
    for name, column in (
        ("fixed_fraction", 0),
        ("preplaced_fraction", 1),
        ("mib_fraction", 2),
        ("cluster_fraction", 3),
        ("boundary_fraction", 4),
    ):
        flags[name] = float(np.any(constraints[:, column : column + 1] != 0, axis=1).mean())
    return flags


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_indices(
    indices: list[int],
    reports: dict[str, dict[int, dict]],
) -> dict:
    result = {"cases": len(indices)}
    for arm, rows in reports.items():
        result[arm] = {
            metric: mean([float(rows[index][metric]) for index in indices])
            for metric in METRICS
            if metric in rows[indices[0]]
        }
    if "flat_dense" in reports and "global_slot" in reports:
        deltas = [
            float(reports["global_slot"][index]["legalized_hard_cost"])
            - float(reports["flat_dense"][index]["legalized_hard_cost"])
            for index in indices
        ]
        result["global_minus_flat_hard_cost"] = {
            "mean": mean(deltas),
            "improved_cases": sum(delta < 0 for delta in deltas),
            "relative_delta": mean(deltas)
            / mean(
                [
                    float(reports["flat_dense"][index]["legalized_hard_cost"])
                    for index in indices
                ]
            ),
        }
    return result


def grouped(
    cases: dict[int, dict],
    reports: dict[str, dict[int, dict]],
    key: str,
) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, flags in cases.items():
        groups[str(flags[key])].append(index)
    return {
        name: summarize_indices(sorted(indices), reports)
        for name, indices in sorted(groups.items())
    }


def density_grouped(
    cases: dict[int, dict],
    reports: dict[str, dict[int, dict]],
    key: str,
) -> dict:
    values = np.asarray([float(flags[key]) for flags in cases.values()])
    median = float(np.median(values))
    groups = {
        "low": sorted(
            index for index, flags in cases.items() if float(flags[key]) <= median
        ),
        "high": sorted(
            index for index, flags in cases.items() if float(flags[key]) > median
        ),
    }
    return {
        f"{name}(median={median:.4f})": summarize_indices(indices, reports)
        for name, indices in groups.items()
        if indices
    }


def main() -> None:
    args = parse_args()
    store = PackedCaseStore(args.cache)
    reports = {
        name: read_cases(ROOT / relative)
        for name, relative in ARMS.items()
    }
    shared_indices = sorted(
        set.intersection(*(set(rows) for rows in reports.values()))
    )
    cases = {
        index: case_flags(
            store,
            index,
            int(reports["global_slot"][index]["blocks"]),
        )
        for index in shared_indices
    }
    coverage = {
        key: sum(bool(flags[key]) for flags in cases.values())
        for key in ("fixed", "preplaced", "mib", "cluster", "boundary")
    }
    strata = {
        key: grouped(cases, reports, key)
        for key in ("length", "fixed", "preplaced", "mib", "cluster", "boundary")
    }
    density_strata = {
        key: density_grouped(cases, reports, key)
        for key in (
            "fixed_fraction",
            "preplaced_fraction",
            "mib_fraction",
            "cluster_fraction",
            "boundary_fraction",
        )
    }
    improvements = []
    for key in ("length", "fixed", "cluster", "boundary"):
        for value, row in strata[key].items():
            delta = row.get("global_minus_flat_hard_cost")
            if delta and delta["mean"] is not None:
                improvements.append(
                    {
                        "stratum": key,
                        "value": value,
                        "cases": row["cases"],
                        "relative_delta": delta["relative_delta"],
                        "improved_cases": delta["improved_cases"],
                    }
                )
    improvements.sort(key=lambda item: item["relative_delta"])
    for key, groups in density_strata.items():
        for value, row in groups.items():
            delta = row.get("global_minus_flat_hard_cost")
            if delta and delta["mean"] is not None:
                improvements.append(
                    {
                        "stratum": key,
                        "value": value,
                        "cases": row["cases"],
                        "relative_delta": delta["relative_delta"],
                        "improved_cases": delta["improved_cases"],
                    }
                )
    improvements.sort(key=lambda item: item["relative_delta"])
    best_stratum = improvements[0] if improvements else None
    weakest_stratum = improvements[-1] if improvements else None
    checks = {
        "shared_cases_100": len(shared_indices) == 100,
        "all_reports_have_metrics": all(
            all(metric in next(iter(rows.values())) for metric in METRICS)
            for rows in reports.values()
        ),
        "constraint_coverage_complete": coverage["boundary"] == 100,
        "global_improves_mean": strata["length"]["21-60"]
        .get("global_minus_flat_hard_cost", {})
        .get("mean", 1.0)
        < 0.0,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "official-100 failure stratification; diagnostic, not causal",
        "checks": checks,
        "constraint_coverage": coverage,
        "strata": strata,
        "density_strata": density_strata,
        "global_slot_vs_flat_ranked": improvements,
        "diagnosis": {
            "largest_relative_gain_stratum": best_stratum,
            "weakest_relative_gain_stratum": weakest_stratum,
            "mib_claim": (
                "The parity report has zero MIB violations on ground truth. "
                "That is not zero MIB metadata coverage; report per-case "
                "predicted MIB violations separately."
            ),
            "raw_feasibility": (
                "All current flow arms require legalization; optimize raw "
                "overlap/feasibility only if P3 does not close the gap."
            ),
        },
        "next_gate": (
            "P3 applies the existing Trinity closed-form refiner to the same "
            "global-slot samples, then P4 targets routing only if a trained "
            "sparse arm is still justified."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "coverage": coverage,
                "largest_gain": best_stratum,
                "weakest_gain": weakest_stratum,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

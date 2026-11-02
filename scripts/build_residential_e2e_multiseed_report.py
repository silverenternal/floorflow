"""Aggregate five-seed estimates, paired tests, Holm, and runtime evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

from floorflow.evaluation.statistical_effect_sizes import (
    rank_biserial_correlation,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 20, 21, 22, 23)
OUTPUT = ROOT / "outputs/reports/residential_e2e_multiseed_v1.json"
TABLE = ROOT / "paper/tables/residential_e2e_multiseed_v1.md"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    half = (
        float(student_t.ppf(0.975, len(array) - 1))
        * std
        / np.sqrt(len(array))
    )
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "min": float(array.min()),
        "max": float(array.max()),
        "method": "student_t_interval_across_five_seeds",
    }


def _refiner_path(seed: int) -> Path:
    if seed == 19:
        return ROOT / "outputs/reports/residential_predicted_refiner_v1.json"
    return ROOT / (
        f"outputs/reports/residential_predicted_refiner_seed{seed}_v1.json"
    )


def _interaction_path(seed: int) -> Path:
    if seed == 19:
        return ROOT / "outputs/reports/residential_interaction_policy_v11.json"
    return ROOT / (
        f"outputs/reports/residential_interaction_policy_seed{seed}_v1.json"
    )


def _paired_test(rows: list[dict], left: str, right: str, metric: str) -> dict:
    differences = np.asarray(
        [row[left][metric] - row[right][metric] for row in rows],
        dtype=np.float64,
    )
    nonzero = differences[np.abs(differences) > 1.0e-12]
    lower_is_better = metric == "ged"
    test = (
        wilcoxon(
            nonzero,
            alternative="less" if lower_is_better else "greater",
            zero_method="wilcox",
        )
        if nonzero.size
        else None
    )
    return {
        "differences": differences,
        "mean_delta": float(differences.mean()),
        "median_delta": float(np.median(differences)),
        "p_value": float(test.pvalue) if test is not None else 1.0,
        "rank_biserial": float(rank_biserial_correlation(differences)),
        "nonzero_pairs": int(nonzero.size),
        "lower_is_better": lower_is_better,
    }


def _holm(results: dict[str, dict], alpha: float = 0.05) -> None:
    ordered = sorted(results, key=lambda name: results[name]["p_value"])
    count = len(ordered)
    previous = 0.0
    for index, name in enumerate(ordered):
        adjusted = min(
            1.0,
            max(
                previous,
                results[name]["p_value"] * count / (index + 1),
            ),
        )
        results[name]["holm_adjusted_p"] = adjusted
        results[name]["holm_significant"] = adjusted <= alpha
        previous = adjusted


def main() -> None:
    refiner_rows = []
    for seed in SEEDS:
        payload = _read(_refiner_path(seed))
        refiner_rows.append(
            {
                "seed": seed,
                "status": payload["status"],
                "room_iou": payload["best"]["room_macro_iou"],
                "room_iou_delta": payload["paired"]["room_macro_iou"][
                    "mean_delta"
                ],
                "wall_f1": payload["best"]["wall_f1"],
                "opening_f1": payload["best"]["opening_f1"],
                "validity": payload["best"]["validity"],
                "watertight": payload["best"]["watertight"],
                "room_closure_rate": payload["best"]["room_closure_rate"],
            }
        )
    refiner_intervals = {
        key: _interval([float(row[key]) for row in refiner_rows])
        for key in (
            "room_iou",
            "room_iou_delta",
            "wall_f1",
            "opening_f1",
            "validity",
            "watertight",
            "room_closure_rate",
        )
    }
    interaction_rows = []
    for seed in SEEDS:
        payload = _read(_interaction_path(seed))
        delta = payload["comparisons"]["interaction_vs_one_step"]
        interaction_rows.append(
            {
                "seed": seed,
                "status": payload["status"],
                "delta": delta["mean_delta"],
                "ci_low": delta["ci_low"],
                "ci_high": delta["ci_high"],
                "hard_feasible": payload["checks"][
                    "interaction_hard_feasible"
                ],
                "legal": payload["checks"]["interaction_selected_legal"],
                "invalid_mass_zero": payload["checks"][
                    "interaction_invalid_mass_zero"
                ],
            }
        )
    interaction_interval = _interval(
        [row["delta"] for row in interaction_rows]
    )
    cagn = _read(ROOT / "outputs/reports/residential_cagn_multiseed_v1.json")
    graph_rows = _read(
        ROOT / "outputs/reports/residential_graph_refinement_exact_rows_v1.json"
    )["rows"]
    graph_primary_rows = [
        {
            "refined": row["metrics"]["direct_tiled"],
            "baseline": row["metrics"]["wall_wall_primary"],
        }
        for row in graph_rows
    ]
    external_rows = _read(
        ROOT / "outputs/reports/residential_external_comparison_rows_v1.json"
    )["rows"]
    primary = {
        "graph_node_f1": _paired_test(
            graph_primary_rows,
            "refined",
            "baseline",
            "node_f1",
        ),
        "graph_edge_overlap": _paired_test(
            graph_primary_rows,
            "refined",
            "baseline",
            "edge_overlap",
        ),
        "graph_ged": _paired_test(
            graph_primary_rows,
            "refined",
            "baseline",
            "ged",
        ),
        "external_room_iou": _paired_test(
            external_rows,
            "floorflow",
            "raster2seq",
            "room_macro_iou",
        ),
        "external_node_f1": _paired_test(
            external_rows,
            "floorflow",
            "raster2seq",
            "node_f1",
        ),
        "external_edge_overlap": _paired_test(
            external_rows,
            "floorflow",
            "raster2seq",
            "edge_overlap",
        ),
        "external_ged": _paired_test(
            external_rows,
            "floorflow",
            "raster2seq",
            "ged",
        ),
    }
    for result in primary.values():
        result.pop("differences", None)
    _holm(primary)
    predicted_edit = _read(
        ROOT / "outputs/reports/residential_predicted_edit_v1.json"
    )
    cagn_stress = _read(
        ROOT / "outputs/reports/residential_cagn_stress_v1.json"
    )
    control_robustness = _read(
        ROOT / "outputs/reports/fp_control_robustness_val191_v1.json"
    )
    runtime = _read(
        ROOT / "outputs/reports/residential_e2e_runtime_benchmark_v1.json"
    )
    interaction_stable = all(row["status"] == "passed" for row in interaction_rows)
    interaction_hard_exact = all(
        row["hard_feasible"] and row["legal"] and row["invalid_mass_zero"]
        for row in interaction_rows
    )
    checks = {
        "refiner_five_seeds": len(refiner_rows) == 5,
        "refiner_all_seed_gates_passed": all(
            row["status"] == "passed" for row in refiner_rows
        ),
        "refiner_seed_room_iou_ci_positive": (
            refiner_intervals["room_iou_delta"]["ci_low"] > 0.0
        ),
        "refiner_validity_exact_all_seeds": all(
            row["validity"] == 1.0 for row in refiner_rows
        ),
        "cagn_five_seeds_passed": cagn["status"] == "passed",
        "cagn_hard_mask_exact": cagn["checks"]["all_hard_masks_exact"],
        "cagn_ece_at_most_0_05": cagn["seed_intervals"]["ece"]["mean"] <= 0.05,
        "interaction_hard_guards_exact": interaction_hard_exact,
        "interaction_stable_for_promotion": interaction_stable,
        "predicted_edit_claim_passed": predicted_edit["status"] == "passed",
        "cagn_d4_and_permutation_passed": (
            cagn_stress["status"] == "passed"
            and cagn_stress["checks"]["permutation_logits_stable_at_1e_5"]
            and cagn_stress["checks"]["d4_ndcg_drop_at_most_0_05"]
        ),
        "missing_control_limitation_reported": (
            control_robustness["checks"]["mild_watertight_floor"] is False
        ),
        "all_primary_endpoints_holm_significant": all(
            value["holm_significant"] for value in primary.values()
        ),
        "runtime_report_complete": runtime["status"] == "complete",
    }
    report = {
        "schema_version": "residential-e2e-multiseed-v1",
        "status": (
            "passed"
            if all(
                value
                for name, value in checks.items()
                if name != "interaction_stable_for_promotion"
            )
            else "failed"
        ),
        "scope": "residential_floorplan_only_validation",
        "seeds": list(SEEDS),
        "test_opened": False,
        "refiner": {
            "rows": refiner_rows,
            "intervals": refiner_intervals,
        },
        "interaction": {
            "rows": interaction_rows,
            "interval": interaction_interval,
            "promotion": "rejected_unstable" if not interaction_stable else "passed",
        },
        "cagn": {
            "status": cagn["status"],
            "seed_intervals": cagn["seed_intervals"],
        },
        "primary_endpoints": primary,
        "predicted_instruction": {
            "status": predicted_edit["status"],
            "node_f1": predicted_edit["paired_refinement"][
                "refined_vs_baseline_node_f1"
            ],
            "edge_overlap": predicted_edit["paired_refinement"][
                "refined_vs_baseline_edge_overlap"
            ],
            "ged_negative_diagnostic": predicted_edit["paired_refinement"][
                "refined_vs_baseline_ged"
            ],
        },
        "stress": {
            "cagn": cagn_stress,
            "control_robustness": control_robustness,
            "missing_control_claim_enabled": False,
            "ood_claim_enabled": False,
        },
        "efficiency": runtime,
        "checks": checks,
    }
    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
            default=lambda value: value.item(),
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Residential End-to-End Five-Seed Study",
        "",
        "| Seed | Refiner gate | Room IoU | Room IoU gain | Validity | Watertight |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in refiner_rows:
        lines.append(
            f"| {row['seed']} | {row['status']} | {row['room_iou']:.6f} | "
            f"{row['room_iou_delta']:+.6f} | {row['validity']:.6f} | "
            f"{row['watertight']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Endpoint | Delta | Wilcoxon p | Holm p | Rank-biserial |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, value in primary.items():
        lines.append(
            f"| {name} | {value['mean_delta']:+.6f} | "
            f"{value['p_value']:.3e} | {value['holm_adjusted_p']:.3e} | "
            f"{value['rank_biserial']:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Interaction promotion: **{report['interaction']['promotion']}**.",
            "",
            (
                f"Refiner latency: {runtime['refiner_latency_ms_per_case']:.3f} ms/case; "
                f"graph readout: {runtime['graph_latency_ms_per_case']:.3f} ms/case; "
                f"extra neural NFE: {runtime['extra_neural_nfe']}."
            ),
        ]
    )
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"report": str(OUTPUT), "checks": checks},
            indent=2,
            default=lambda value: value.item(),
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

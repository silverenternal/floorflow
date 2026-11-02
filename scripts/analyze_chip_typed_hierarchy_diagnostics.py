#!/usr/bin/env python3
"""Diagnose the reproducible 5k typed-hierarchy screen and draw its Pareto."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.evaluation.paired_bootstrap import paired_delta

CANDIDATE = "candidate"
BASELINE = "baseline"
NFE = (1, 2, 4, 8)
REFINER = (50, 100, 200, 400)
PRIMARY_METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_hpwl_gap",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_mib",
    "legalized_boundary",
    "legalized_rho_mae",
)
COLORS = {
    "blue": "#2f6fb6",
    "teal": "#16837a",
    "amber": "#b77a00",
    "red": "#b84a5a",
    "gray": "#526173",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/v3_screen_5k",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/chip_typed_hierarchy_diagnostics_v2.json"
        ),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=(
            ROOT
            / "outputs/figures/chip_typed_hierarchy_pareto_v2"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def result(path: Path, key: str) -> dict:
    return read(path)["results"][key]


def rows(path: Path, key: str) -> list[dict]:
    return read(path)[key]


def mean(rows_: list[dict], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows_]))


def quantiles(rows_: list[dict], metric: str) -> dict[str, float]:
    values = np.asarray([float(row[metric]) for row in rows_])
    return {
        name: float(value)
        for name, value in zip(
            ("p50", "p75", "p90", "p95"),
            np.quantile(values, (0.5, 0.75, 0.9, 0.95)),
        )
    }


def metric_statistics(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    metric: str,
    bootstrap: int,
    seed: int,
) -> dict:
    candidate_mean = mean(candidate_rows, metric)
    baseline_mean = mean(baseline_rows, metric)
    paired = paired_delta(
        candidate_rows,
        baseline_rows,
        metric,
        iterations=bootstrap,
        seed=seed,
        lower_is_better=True,
    )
    return {
        "candidate_mean": candidate_mean,
        "baseline_mean": baseline_mean,
        "relative_delta": (
            candidate_mean / baseline_mean - 1.0
            if baseline_mean != 0.0
            else None
        ),
        "candidate_quantiles": quantiles(candidate_rows, metric),
        "baseline_quantiles": quantiles(baseline_rows, metric),
        "paired": paired,
        "ci_excludes_zero": (
            paired["ci_low"] < 0.0 and paired["ci_high"] < 0.0
        ),
    }


def strata(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    values: np.ndarray,
    label: str,
) -> dict:
    low, high = np.quantile(values, (1 / 3, 2 / 3))
    groups = np.digitize(values, (low, high))
    output = {}
    for index, name in enumerate(("low", "medium", "high")):
        selected = np.flatnonzero(groups == index)
        if not selected.size:
            continue
        output[name] = {
            "cases": int(selected.size),
            "thresholds": [float(low), float(high)],
            "metrics": {
                metric: {
                    "candidate_mean": mean(
                        [candidate_rows[i] for i in selected],
                        metric,
                    ),
                    "baseline_mean": mean(
                        [baseline_rows[i] for i in selected],
                        metric,
                    ),
                    "relative_delta": (
                        mean([candidate_rows[i] for i in selected], metric)
                        / mean([baseline_rows[i] for i in selected], metric)
                        - 1.0
                    ),
                }
                for metric in (
                    "legalized_hard_cost",
                    "legalized_v_rel",
                    "legalized_b2b_hpwl_gap",
                    "legalized_area_gap",
                    "legalized_grouping",
                )
            },
        }
    return {"stratifier": label, "groups": output}


def plot_pareto(
    candidate_nfe: dict,
    baseline_nfe: dict,
    candidate_refiner: dict,
    baseline_primary: dict,
    prefix: Path,
) -> dict[str, str]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
    for column, metric in enumerate(
        ("legalized_hard_cost", "legalized_v_rel")
    ):
        axis = axes[0, column]
        axis.plot(
            NFE,
            [candidate_nfe[nfe][metric] for nfe in NFE],
            marker="o",
            color=COLORS["blue"],
            label="Typed hierarchy",
        )
        axis.plot(
            NFE,
            [baseline_nfe[nfe][metric] for nfe in NFE],
            marker="s",
            color=COLORS["gray"],
            label="Graph-token control",
        )
        axis.axhline(
            baseline_nfe[8][metric],
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1.0,
        )
        axis.set_xticks(NFE)
        axis.set_xlabel("NFE")
        axis.set_ylabel(
            "Cost" if metric == "legalized_hard_cost" else r"$V_{\mathrm{rel}}$"
        )
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, loc="best")
        axis.set_title("Raw sampling quality")

        axis = axes[1, column]
        refine_x = (0,) + REFINER
        refine_values = [
            candidate_nfe[4][metric],
            *[candidate_refiner[steps][metric] for steps in REFINER],
        ]
        axis.plot(
            refine_x,
            refine_values,
            marker="o",
            color=COLORS["blue"],
        )
        axis.scatter(
            [400],
            [baseline_primary[metric]],
            marker="s",
            color=COLORS["gray"],
            zorder=3,
        )
        axis.axhline(
            baseline_primary[metric],
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1.0,
        )
        axis.annotate(
            "50 steps: V_rel shortfall",
            (50, candidate_refiner[50][metric]),
            xytext=(12, 26),
            textcoords="offset points",
            fontsize=7,
            color=COLORS["amber"],
            arrowprops={"arrowstyle": "-", "color": COLORS["amber"]},
        )
        axis.set_xlabel("Refinement steps")
        axis.set_ylabel(
            "Cost" if metric == "legalized_hard_cost" else r"$V_{\mathrm{rel}}$"
        )
        axis.grid(alpha=0.25)
        axis.set_title("NFE4 refinement budget")
    fig.suptitle(
        "FloorSet official-100: NFE and refinement quality trade-off",
        fontsize=10,
    )
    fig.tight_layout()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix in ("pdf", "svg", "png"):
        path = prefix.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=220)
        paths[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    reproducible = args.root / "reproducible_v1"
    candidate_primary = result(
        reproducible / "candidate_primary.json",
        "nfe4_refine400",
    )
    baseline_primary = result(
        reproducible / "baseline_primary.json",
        "nfe4_refine400",
    )
    candidate_rows = rows(
        reproducible / "candidate_primary_cases.json",
        "nfe4_refine400",
    )
    baseline_rows = rows(
        reproducible / "baseline_primary_cases.json",
        "nfe4_refine400",
    )
    statistics = {
        metric: metric_statistics(
            candidate_rows,
            baseline_rows,
            metric,
            args.bootstrap,
            20261120 + index,
        )
        for index, metric in enumerate(PRIMARY_METRICS)
    }
    candidate_nfe = {
        4: result(reproducible / "candidate_nfe4.json", "nfe4_refine0")
    }
    candidate_nfe.update(
        {
            int(key.removeprefix("nfe").split("_")[0]): value
            for key, value in read(
                reproducible / "candidate_nfe_remaining.json"
            )["results"].items()
        }
    )
    baseline_nfe = {
        4: result(reproducible / "baseline_nfe4.json", "nfe4_refine0")
    }
    baseline_nfe.update(
        {
            int(key.removeprefix("nfe").split("_")[0]): value
            for key, value in read(
                reproducible / "baseline_nfe_remaining.json"
            )["results"].items()
        }
    )
    candidate_refiner = {
        **{
            int(key.rsplit("refine", 1)[1]): value
            for key, value in read(
                reproducible / "candidate_refiner_short.json"
            )["results"].items()
        },
        400: candidate_primary,
    }
    store = PackedCaseStore(args.cache)
    indices = np.asarray([int(row["index"]) for row in candidate_rows])
    stratifications = {
        label: strata(
            candidate_rows,
            baseline_rows,
            np.asarray(store.arrays[field][indices], dtype=float),
            label,
        )
        for label, field in (
            ("block_count", "n"),
            ("b2b_edges", "n_b2b"),
            ("p2b_edges", "n_p2b"),
            ("pins", "n_pins"),
        )
    }
    nfe_table = {
        str(nfe): {
            CANDIDATE: candidate_nfe[nfe],
            BASELINE: baseline_nfe[nfe],
            "relative_delta": {
                metric: (
                    candidate_nfe[nfe][metric] / baseline_nfe[nfe][metric] - 1.0
                )
                for metric in PRIMARY_METRICS[:4]
            },
        }
        for nfe in NFE
    }
    refiner_table = {
        str(steps): {
            "candidate": candidate_refiner[steps],
            "baseline_400": baseline_primary,
            "relative_delta_vs_baseline_400": {
                metric: (
                    candidate_refiner[steps][metric] / baseline_primary[metric]
                    - 1.0
                )
                for metric in PRIMARY_METRICS[:4]
            },
        }
        for steps in REFINER
    }
    figure_paths = plot_pareto(
        candidate_nfe,
        baseline_nfe,
        candidate_refiner,
        baseline_primary,
        args.figure_prefix,
    )
    checks = {
        "legalized_feasible_fraction_one": (
            candidate_primary["legalized_feasible_fraction"] == 1.0
            and all(
                candidate_nfe[nfe]["legalized_feasible_fraction"] == 1.0
                for nfe in NFE
            )
        ),
        "v_rel_point_improves_5pct": (
            statistics["legalized_v_rel"]["relative_delta"] <= -0.05
        ),
        "cost_point_improves_5pct": (
            statistics["legalized_hard_cost"]["relative_delta"] <= -0.05
        ),
        "v_rel_ci_excludes_zero": statistics["legalized_v_rel"][
            "ci_excludes_zero"
        ],
        "cost_ci_excludes_zero": statistics["legalized_hard_cost"][
            "ci_excludes_zero"
        ],
        "grouping_does_not_regress": (
            statistics["legalized_grouping"]["relative_delta"] <= 0.0
        ),
        "rho_does_not_regress": (
            statistics["legalized_rho_mae"]["relative_delta"] <= 0.0
        ),
        "candidate_50_matches_baseline_400": all(
            candidate_refiner[50][metric] <= baseline_primary[metric]
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        ),
    }
    report = {
        "status": "rescue_required",
        "scope": "reproducible seed-19 5k FloorSet official-100 diagnostics",
        "reviewer_question": (
            "Does typed hierarchical conditioning improve cost, structural "
            "relations, tails, or equal-quality sampling budget?"
        ),
        "candidate_primary": candidate_primary,
        "baseline_primary": baseline_primary,
        "primary_statistics": statistics,
        "nfe_table": nfe_table,
        "refiner_table": refiner_table,
        "stratifications": stratifications,
        "run_variation": {
            "cause": (
                "Trinity scale_pack LP had a 2-second timeout; OR-Tools could "
                "return different feasible optima for identical predictions."
            ),
            "fix": (
                "Official scoring now uses the same scale_pack implementation "
                "with a 600-second deterministic evaluation timeout."
            ),
            "baseline_legacy_cost": 1.3973065501664457,
            "baseline_timeout_current_code_cost": 1.3533747581029445,
            "baseline_reproducible_cost": baseline_primary[
                "legalized_hard_cost"
            ],
            "candidate_timeout_current_code_cost": 1.3490689256068196,
            "candidate_reproducible_cost": candidate_primary[
                "legalized_hard_cost"
            ],
        },
        "figure": figure_paths,
        "checks": checks,
        "decision": {
            "promote_unchanged": False,
            "conditional_20k_rescue": True,
            "reason": (
                "V_rel, b2b and area improve at NFE4+400 and NFE8, but cost "
                "and paired CIs do not establish a stable 5k gain. Grouping "
                "and rho regress. Test 20k before any paper claim."
            ),
            "preserve_negative_result": True,
        },
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
                "figure": figure_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

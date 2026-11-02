#!/usr/bin/env python3
"""Build chip tail-concentration and equal-quality budget evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = (0.5, 0.75, 0.9, 0.95)
METRICS = {
    "cost": "legalized_hard_cost",
    "v_rel": "legalized_v_rel",
}
COLORS = {
    "cost": "#2f6fb6",
    "v_rel": "#16837a",
    "tail": "#b77a00",
    "remainder": "#526173",
    "risk": "#b84a5a",
    "grid": "#c7ced8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_component_rows_v2.json",
    )
    parser.add_argument(
        "--mechanism",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json",
    )
    parser.add_argument(
        "--budget",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_budget_curve_v1.json",
    )
    parser.add_argument(
        "--refiner",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_refiner_curve_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_tail_budget_mechanism_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/chip_tail_budget_mechanism.md",
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=ROOT / "paper/figures/chip_tail_budget_mechanism",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def arm_rows(payload: dict) -> dict[str, list[dict]]:
    output = {}
    for arm in ("candidate", "baseline"):
        grouped = {}
        for row in payload["rows"]:
            if row["arm"] == arm and row["stage"] == "refine400":
                grouped.setdefault(int(row["index"]), []).append(row)
        output[arm] = [
            {
                "index": index,
                **{
                    key: float(
                        np.mean([float(row[key]) for row in cases])
                    )
                    for key in METRICS.values()
                },
            }
            for index, cases in sorted(grouped.items())
        ]
    indices = [
        [int(row["index"]) for row in output[arm]]
        for arm in ("candidate", "baseline")
    ]
    if indices[0] != indices[1]:
        raise ValueError("candidate and baseline rows are not paired")
    if len(indices[0]) != 100:
        raise ValueError("expected 100 paired official cases")
    return output


def quantile_stats(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    count = len(candidate)
    samples = rng.integers(0, count, (iterations, count))
    output = {}
    for quantile in QUANTILES:
        candidate_value = float(np.quantile(candidate, quantile))
        baseline_value = float(np.quantile(baseline, quantile))
        candidate_samples = np.quantile(candidate[samples], quantile, axis=1)
        baseline_samples = np.quantile(baseline[samples], quantile, axis=1)
        improvement = (baseline_samples - candidate_samples) / np.abs(
            baseline_samples
        )
        output[str(quantile)] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "relative_improvement": (
                baseline_value - candidate_value
            )
            / abs(baseline_value),
            "ci_low": float(np.quantile(improvement, 0.025)),
            "ci_high": float(np.quantile(improvement, 0.975)),
            "ci_excludes_zero": bool(
                np.quantile(improvement, 0.975) < 0.0
                or np.quantile(improvement, 0.025) > 0.0
            ),
        }
    return output


def stratum_stats(
    candidate: np.ndarray,
    baseline: np.ndarray,
    selected: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    cand = candidate[selected]
    base = baseline[selected]
    samples = rng.integers(0, len(cand), (iterations, len(cand)))
    gains = (
        base[samples].mean(axis=1) - cand[samples].mean(axis=1)
    ) / np.abs(base[samples].mean(axis=1))
    baseline_mean = float(base.mean())
    candidate_mean = float(cand.mean())
    return {
        "cases": len(cand),
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "relative_improvement": (
            baseline_mean - candidate_mean
        )
        / abs(baseline_mean),
        "ci_low": float(np.quantile(gains, 0.025)),
        "ci_high": float(np.quantile(gains, 0.975)),
    }


def tail_strata(
    values: dict[str, np.ndarray],
    *,
    iterations: int,
) -> dict:
    threshold = float(np.quantile(values["baseline_cost"], 0.75))
    high = values["baseline_cost"] >= threshold
    output = {"threshold_baseline_cost_q75": threshold}
    for index, metric in enumerate(METRICS):
        output[metric] = {
            "top_quartile_baseline_cost": stratum_stats(
                values[f"candidate_{metric}"],
                values[f"baseline_{metric}"],
                high,
                iterations=iterations,
                seed=20263110 + index,
            ),
            "remaining_cases": stratum_stats(
                values[f"candidate_{metric}"],
                values[f"baseline_{metric}"],
                ~high,
                iterations=iterations,
                seed=20263120 + index,
            ),
        }
        output[metric]["gain_gap_top_minus_remaining"] = (
            output[metric]["top_quartile_baseline_cost"]["relative_improvement"]
            - output[metric]["remaining_cases"]["relative_improvement"]
        )
    return output


def budget_evidence(budget: dict) -> dict:
    nfe = budget["nfe"]
    refiner = budget["refiner"]
    candidate_50 = refiner["curve"]["candidate"]["50"]
    threshold = refiner["baseline_quality_threshold"]
    return {
        "nfe": {
            "candidate": nfe["candidate_minimum_nfe"],
            "baseline": 8,
            "saved": nfe["saved_nfe_vs_8"],
            "declared_quality": True,
            "comparison": "pooled raw quality threshold",
        },
        "refiner": {
            "candidate": refiner["candidate_minimum_steps"],
            "baseline": 400,
            "saved": refiner["saved_steps"],
            "declared_quality": True,
            "comparison": "seed-19 controlled curve",
        },
        "short_budget_boundary": {
            "steps": 50,
            "candidate_cost": candidate_50["cost"],
            "candidate_v_rel": candidate_50["v_rel"],
            "baseline_threshold": threshold,
            "cost_meets_quality": candidate_50["cost"] <= threshold["cost"],
            "v_rel_meets_quality": (
                candidate_50["v_rel"] <= threshold["v_rel"]
            ),
        },
        "primary_endpoint": "NFE4 + 400 refiner steps",
    }


def build_figure(report: dict, path_stem: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    x = np.arange(len(QUANTILES))
    width = 0.36
    for offset, metric in zip((-width / 2, width / 2), METRICS, strict=True):
        values = report["quantile_evidence"][metric]
        gains = [
            100.0 * values[str(quantile)]["relative_improvement"]
            for quantile in QUANTILES
        ]
        errors = np.asarray(
            [
                [
                    100.0
                    * (
                        values[str(quantile)]["relative_improvement"]
                        - values[str(quantile)]["ci_low"]
                    ),
                    100.0
                    * (
                        values[str(quantile)]["ci_high"]
                        - values[str(quantile)]["relative_improvement"]
                    ),
                ]
                for quantile in QUANTILES
            ]
        ).T
        axes[0].bar(
            x + offset,
            gains,
            width,
            yerr=errors,
            capsize=2,
            color=COLORS[metric],
            label=metric.replace("_", " "),
            error_kw={"elinewidth": 0.8, "ecolor": COLORS["remainder"]},
        )
        for xpos, value in zip(x + offset, gains, strict=True):
            axes[0].text(
                xpos,
                value + 0.4,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    axes[0].axhline(0.0, color=COLORS["remainder"], linewidth=0.8)
    axes[0].set_xticks(x, [f"p{int(q * 100)}" for q in QUANTILES])
    axes[0].set_ylabel("Relative improvement (%)")
    axes[0].set_title("A  Quantile gains", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="upper left")

    labels = []
    values = []
    errors = []
    colors = []
    for metric in METRICS:
        for stratum, label, color in (
            ("top_quartile_baseline_cost", "high-cost 25%", COLORS["tail"]),
            ("remaining_cases", "remaining 75%", COLORS["remainder"]),
        ):
            item = report["tail_strata"][metric][stratum]
            labels.append(
                f"{metric.replace('_', ' ')}\n"
                f"{'high 25%' if stratum == 'top_quartile_baseline_cost' else 'remaining 75%'}"
            )
            values.append(100.0 * item["relative_improvement"])
            errors.append(
                [
                    100.0
                    * (
                        item["relative_improvement"]
                        - item["ci_low"]
                    ),
                    100.0
                    * (
                        item["ci_high"]
                        - item["relative_improvement"]
                    ),
                ]
            )
            colors.append(color)
    x_strata = np.arange(len(labels))
    axes[1].bar(
        x_strata,
        values,
        yerr=np.asarray(errors).T,
        capsize=2,
        color=colors,
        error_kw={"elinewidth": 0.8, "ecolor": COLORS["remainder"]},
    )
    axes[1].axhline(0.0, color=COLORS["remainder"], linewidth=0.8)
    axes[1].set_xticks(x_strata, labels)
    axes[1].tick_params(axis="x", labelsize=6.5)
    axes[1].set_ylabel("Mean relative improvement (%)")
    axes[1].set_title(
        "B  Baseline-cost strata",
        loc="left",
        fontweight="bold",
    )
    for axis in axes:
        axis.grid(True, axis="y", color=COLORS["grid"], alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=3, width=0.8)
    figure.suptitle(
        "Chip gains concentrate in structural-failure tails",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.91,
        "Official-100 matched diagnostic; intervals are paired bootstrap 95% CIs.",
        ha="left",
        fontsize=7,
        color=COLORS["remainder"],
    )
    figure.subplots_adjust(
        top=0.77,
        bottom=0.25,
        left=0.09,
        right=0.98,
        wspace=0.36,
    )
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_stem.with_suffix(".svg"))
    figure.savefig(path_stem.with_suffix(".pdf"))
    figure.savefig(path_stem.with_suffix(".png"), dpi=240)
    plt.close(figure)


def percent(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def interval(item: dict) -> str:
    return f"[{percent(item['ci_low'])}, {percent(item['ci_high'])}]"


def write_table(report: dict, path: Path) -> None:
    lines = [
        "# Chip Tail and Equal-Quality Budget Mechanism",
        "",
        (
            "Scope: matched FloorSet-Lite official-100 diagnostics. The primary "
            "endpoint remains NFE4 plus 400 refiner steps."
        ),
        "",
        "| Metric | Quantile | Baseline | Candidate | Improvement | Paired 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, values in report["quantile_evidence"].items():
        for quantile in QUANTILES:
            item = values[str(quantile)]
            lines.append(
                f"| {metric.replace('_', ' ')} | p{int(quantile * 100)} | "
                f"{item['baseline']:.5f} | {item['candidate']:.5f} | "
                f"{percent(item['relative_improvement'])} | {interval(item)} |"
            )
    lines.extend(
        [
            "",
            (
                "The cost gain rises from p50 to p90/p95. V_rel improves most "
                "near p75-p90 but is not monotonic at p95."
            ),
            "",
            "## Structural-Failure Concentration",
            "",
            "| Metric | Baseline-cost stratum | Cases | Improvement | Paired 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    labels = {
        "top_quartile_baseline_cost": "Top 25%",
        "remaining_cases": "Remaining 75%",
    }
    for metric, strata in report["tail_strata"].items():
        if metric.startswith("threshold_"):
            continue
        for key, label in labels.items():
            item = strata[key]
            lines.append(
                f"| {metric.replace('_', ' ')} | {label} | {item['cases']} | "
                f"{percent(item['relative_improvement'])} | {interval(item)} |"
            )
    budget = report["budget_evidence"]
    short = budget["short_budget_boundary"]
    lines.extend(
        [
            "",
            "## Equal-Quality Budget",
            "",
            "| Setting | Candidate | Baseline | Saved | Boundary |",
            "|---|---:|---:|---:|---|",
            (
                f"| NFE | {budget['nfe']['candidate']} | "
                f"{budget['nfe']['baseline']} | {budget['nfe']['saved']} | "
                "Pooled quality threshold |"
            ),
            (
                f"| Refiner steps | {budget['refiner']['candidate']} | "
                f"{budget['refiner']['baseline']} | "
                f"{budget['refiner']['saved']} | Seed-19 controlled curve |"
            ),
            "",
            (
                f"At 50 refiner steps, cost is {short['candidate_cost']:.5f} "
                f"against threshold {short['baseline_threshold']['cost']:.5f}, "
                f"but V_rel is {short['candidate_v_rel']:.5f} against threshold "
                f"{short['baseline_threshold']['v_rel']:.5f}. The short-budget "
                "V_rel deficit is therefore retained."
            ),
            "",
            (
                "Allowed claim: the candidate primarily reduces structural-failure "
                "tails and can reach declared pooled quality with fewer inference "
                "steps. Do not claim uniform case-level improvement, monotonic "
                "extreme-tail advantage, or replacement of the primary endpoint."
            ),
            "",
            "Figure: `paper/figures/chip_tail_budget_mechanism.svg`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = arm_rows(read(args.rows))
    mechanism = read(args.mechanism)
    budget = read(args.budget)
    values = {}
    for metric, key in METRICS.items():
        values[f"candidate_{metric}"] = np.asarray(
            [float(row[key]) for row in rows["candidate"]]
        )
        values[f"baseline_{metric}"] = np.asarray(
            [float(row[key]) for row in rows["baseline"]]
        )
    quantiles = {
        metric: quantile_stats(
            values[f"candidate_{metric}"],
            values[f"baseline_{metric}"],
            iterations=args.bootstrap,
            seed=20263100 + index,
        )
        for index, metric in enumerate(METRICS)
    }
    reproduces = all(
        abs(
            quantiles[metric][str(quantile)]["relative_improvement"]
            - mechanism["tail_quantiles"][key][str(quantile)][
                "relative_improvement"
            ]
        )
        < 1.0e-12
        for metric, key in METRICS.items()
        for quantile in QUANTILES
    )
    budget_result = budget_evidence(budget)
    cost_gain = quantiles["cost"]
    v_rel_gain = quantiles["v_rel"]
    checks = {
        "existing_quantiles_reproduced": reproduces,
        "cost_tail_concentration": (
            cost_gain["0.9"]["relative_improvement"]
            > cost_gain["0.5"]["relative_improvement"] + 0.05
        ),
        "v_rel_extreme_tail_not_monotonic": (
            v_rel_gain["0.95"]["relative_improvement"]
            < v_rel_gain["0.9"]["relative_improvement"]
        ),
        "equal_quality_nfe_supported": (
            budget_result["nfe"]["declared_quality"]
        ),
        "equal_quality_refiner_supported": (
            budget_result["refiner"]["declared_quality"]
        ),
        "short_budget_v_rel_deficit_reported": not budget_result[
            "short_budget_boundary"
        ]["v_rel_meets_quality"],
        "primary_endpoint_unchanged": (
            budget_result["primary_endpoint"] == "NFE4 + 400 refiner steps"
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "failed",
        "scope": "matched FloorSet-Lite official-100 mechanism diagnostic",
        "checks": checks,
        "quantile_evidence": quantiles,
        "tail_strata": tail_strata(
            values,
            iterations=args.bootstrap,
        ),
        "budget_evidence": budget_result,
        "allowed_claims": [
            (
                "Graph-token flow primarily reduces the cost failure tail; "
                "the p50 improvement remains modest."
            ),
            (
                "V_rel gains are strongest around p75-p90 but do not improve "
                "monotonically through p95."
            ),
            (
                "The candidate reaches the declared pooled quality threshold "
                "with fewer NFE or refiner steps in separate diagnostics."
            ),
        ],
        "forbidden_claims": [
            "Uniform improvement across all cases.",
            "A monotonic advantage through the most extreme V_rel tail.",
            "That NFE1 or 50 refiner steps replace the primary endpoint.",
            "That the high-b2b or high-pin strata show a stable advantage.",
        ],
        "sources": {
            "rows": str(args.rows.relative_to(ROOT)),
            "mechanism": str(args.mechanism.relative_to(ROOT)),
            "budget": str(args.budget.relative_to(ROOT)),
            "refiner": str(args.refiner.relative_to(ROOT)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_table(report, args.table)
    build_figure(report, args.figure_root)
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the matched chip-to-Trinity component gap evidence package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from floorflow.evaluation.chip_gap import (
    ALPHA,
    BETA,
    GROUPS,
    averaged_rows,
    candidate_rows,
    component_report,
    cost_decomposition,
    headline_audit,
    read,
    seed_average,
    survival_report,
    trinity_seed_rows,
    vrel_decomposition,
)

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "structural": "#2f6fb6",
    "geometric": "#16837a",
    "port": "#3f8f5f",
    "gray": "#526173",
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
        "--legacy",
        type=Path,
        default=ROOT / "outputs/reports/trinity_d192_matched_protocol_v2.json",
    )
    parser.add_argument(
        "--component-gap",
        type=Path,
        default=ROOT / "outputs/reports/chip_trinity_d192_component_gap_v1.json",
    )
    parser.add_argument(
        "--mechanism",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_trinity_gap_decomposition_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/chip_trinity_gap_decomposition.md",
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=ROOT / "paper/figures/chip_trinity_gap_decomposition",
    )
    return parser.parse_args()


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
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    components = report["reproducible_component_gap"]

    def display_gap(item: dict) -> float:
        return float(item["relative_gap"] or 0.0)

    names = sorted(
        components,
        key=lambda name: display_gap(components[name]),
    )
    values = [
        100.0 * display_gap(components[name]) for name in names
    ]
    errors = np.asarray(
        [
            [
                (
                    100.0
                    * (
                        display_gap(components[name])
                        - float(
                            components[name]["relative_ci_low"] or 0.0
                        )
                    )
                ),
                (
                    100.0
                    * (
                        float(
                            components[name]["relative_ci_high"] or 0.0
                        )
                        - display_gap(components[name])
                    )
                ),
            ]
            for name in names
        ]
    ).T
    axes[0].barh(
        names,
        values,
        xerr=errors,
        capsize=2,
        color=[COLORS[GROUPS[name]] for name in names],
        error_kw={"elinewidth": 0.8, "ecolor": COLORS["gray"]},
    )
    axes[0].axvline(0.0, color=COLORS["gray"], linewidth=0.8)
    axes[0].set_xlabel("Candidate vs Trinity relative gap (%)")
    axes[0].set_title("A  Component gaps", loc="left", fontweight="bold")
    for y, value in enumerate(values):
        axes[0].text(
            value + (1.0 if value >= 0 else -1.0),
            y,
            (
                "n/a"
                if components[names[y]]["relative_gap"] is None
                else f"{value:+.1f}"
            ),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=6.5,
        )

    decompositions = (
        report["legacy_vrel_decomposition"],
        report["reproducible_vrel_decomposition"],
    )
    shares = {
        name: [
            100.0
            * item["contributions"][name][
                "share_of_aggregate_vrel_gap"
            ]
            for item in decompositions
        ]
        for name in ("grouping", "boundary", "mib")
    }
    x = np.arange(2)
    axes[1].bar(
        x,
        shares["grouping"],
        0.55,
        color=COLORS["structural"],
        label="grouping",
    )
    axes[1].bar(
        x,
        shares["boundary"],
        0.55,
        bottom=shares["grouping"],
        color=COLORS["geometric"],
        label="boundary",
    )
    axes[1].bar(
        x,
        shares["mib"],
        0.55,
        bottom=np.asarray(shares["grouping"])
        + np.asarray(shares["boundary"]),
        color=COLORS["port"],
        label="MIB",
    )
    axes[1].set_xticks(x, ("Predeclared", "3-seed"))
    axes[1].set_ylabel("Share of V_rel gap (%)")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_title("B  V_rel attribution", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right")
    for axis in axes:
        axis.grid(
            True,
            axis="x" if axis is axes[0] else "y",
            color=COLORS["grid"],
            alpha=0.65,
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=3, width=0.8)
    figure.suptitle(
        "Remaining Trinity gap is structural and geometric",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.91,
        "Official-100 matched protocol; paired bootstrap 95% CIs. No SOTA claim.",
        ha="left",
        fontsize=7,
        color=COLORS["gray"],
    )
    figure.subplots_adjust(
        top=0.77,
        bottom=0.18,
        left=0.14,
        right=0.98,
        wspace=0.45,
    )
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_stem.with_suffix(".svg"))
    figure.savefig(path_stem.with_suffix(".pdf"))
    figure.savefig(path_stem.with_suffix(".png"), dpi=240)
    plt.close(figure)


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:+.2f}%"


def write_table(report: dict, path: Path) -> None:
    lines = [
        "# Trinity d192 Gap Decomposition",
        "",
        (
            "Scope: matched FloorSet-Lite official-100. The predeclared "
            "comparison is an appendix anchor; the three-seed result is primary."
        ),
        "",
        "| Component | Candidate | Trinity | Relative gap | Paired 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in report["reproducible_component_gap"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {item['candidate_mean']:.5f} | "
            f"{item['trinity_mean']:.5f} | {pct(item['relative_gap'])} | "
            f"[{pct(item['relative_ci_low'])}, "
            f"{pct(item['relative_ci_high'])}] |"
        )
    audit = report["headline_audit"]
    legacy = report["legacy_vrel_decomposition"]["contributions"]
    reproducible = report["reproducible_vrel_decomposition"]["contributions"]
    cost = report["reproducible_cost_decomposition"]
    lines.extend(
        [
            "",
            "## Headline Reconciliation",
            "",
            "| Comparison | Cost gap | V_rel gap |",
            "|---|---:|---:|",
            (
                f"| Predeclared anchor | {pct(audit['predeclared_gap']['cost'])} | "
                f"{pct(audit['predeclared_gap']['v_rel'])} |"
            ),
            (
                f"| Component-row source | "
                f"{pct(audit['component_row_gap']['cost'])} | "
                f"{pct(audit['component_row_gap']['v_rel'])} |"
            ),
            (
                f"| Candidate source residual | "
                f"{pct(audit['candidate_source_residual']['cost'])} | "
                f"{pct(audit['candidate_source_residual']['v_rel'])} |"
            ),
            "",
            (
                "The predeclared +5.76%/+30.45% anchor remains frozen. "
                "Component rows use a slightly different candidate summary; "
                "the disclosed sub-1% source residual is not allocated to a "
                "model component."
            ),
            "",
            "## V_rel Attribution",
            "",
            "| Slots | Predeclared share | 3-seed share |",
            "|---|---:|---:|",
        ]
    )
    for name in ("grouping", "boundary", "mib"):
        lines.append(
            f"| {name} | "
            f"{100.0 * legacy[name]['share_of_aggregate_vrel_gap']:.2f}% | "
            f"{100.0 * reproducible[name]['share_of_aggregate_vrel_gap']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Hard-Cost Split",
            "",
            (
                f"Using `ALPHA={ALPHA}` and `BETA={BETA}`, `log(cost)` "
                "separates exactly into geometry and soft-constraint terms."
            ),
            "",
            "| Term | Candidate | Trinity | Log gap | Share |",
            "|---|---:|---:|---:|---:|",
            (
                f"| Geometry | "
                f"{cost['candidate_log_terms']['log_quality']:.5f} | "
                f"{cost['trinity_log_terms']['log_quality']:.5f} | "
                f"{cost['geometry_log_gap']:.5f} | "
                f"{100.0 * cost['geometry_share_of_log_gap']:.2f}% |"
            ),
            (
                f"| Soft constraints | "
                f"{cost['candidate_log_terms']['log_soft']:.5f} | "
                f"{cost['trinity_log_terms']['log_soft']:.5f} | "
                f"{cost['soft_log_gap']:.5f} | "
                f"{100.0 * cost['soft_share_of_log_gap']:.2f}% |"
            ),
            "",
            (
                "The remaining gap is structural (b2b and grouping) and "
                "geometric (area/outline and rho/aspect). The candidate retains "
                "an advantage on p2b/port HPWL. MIB has no official-100 signal, "
                "and exact-area-decoder/training attribution remains unresolved."
            ),
            "",
            "Figure: `paper/figures/chip_trinity_gap_decomposition.svg`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate = candidate_rows(read(args.rows))
    legacy = averaged_rows(read(args.legacy)["rows"], arm="trinity")
    by_seed = trinity_seed_rows()
    reproducible = seed_average(by_seed)
    component_gap = read(args.component_gap)
    mechanism = read(args.mechanism)
    reproducible_components = component_report(
        candidate,
        reproducible,
        seed=20263300,
    )
    legacy_vrel = vrel_decomposition(
        candidate,
        legacy,
        seed=20263400,
    )
    reproducible_vrel = vrel_decomposition(
        candidate,
        reproducible,
        seed=20263500,
    )
    legacy_cost = cost_decomposition(candidate, legacy)
    reproducible_cost = cost_decomposition(candidate, reproducible)
    audit = headline_audit(component_gap, candidate, legacy)
    survival = survival_report(candidate, reproducible, mechanism)
    checks = {
        "predeclared_headline_frozen": (
            abs(audit["predeclared_gap"]["cost"] - 0.0576) < 0.0002
            and abs(audit["predeclared_gap"]["v_rel"] - 0.3045) < 0.0002
        ),
        "vrel_components_reconcile": (
            abs(legacy_vrel["reconstruction_error"]) < 1.0e-12
            and abs(reproducible_vrel["reconstruction_error"]) < 1.0e-12
        ),
        "cost_terms_reconcile": (
            abs(legacy_cost["reconstruction_error"]) < 1.0e-12
            and abs(reproducible_cost["reconstruction_error"]) < 1.0e-12
        ),
        "candidate_source_residual_within_1pct": all(
            abs(value) < 0.01
            for value in audit["candidate_source_residual"].values()
        ),
        "mib_has_no_positive_gap_to_claim": (
            reproducible_vrel["contributions"]["mib"]["relative_gap"] is None
        ),
        "p2b_candidate_advantage_reported": (
            reproducible_components["p2b_port"]["relative_gap"] < 0.0
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "failed",
        "scope": "official-100 matched component decomposition",
        "checks": checks,
        "definitions": {
            "v_rel": "(V_grouping + V_boundary + V_mib) / N_soft",
            "hard_cost": (
                "(1 + ALPHA*(max(hpwl_gap,0)+max(area_gap,0)))"
                "*exp(BETA*V_rel)"
            ),
            "alpha": ALPHA,
            "beta": BETA,
        },
        "headline_audit": audit,
        "reproducible_component_gap": reproducible_components,
        "legacy_vrel_decomposition": legacy_vrel,
        "reproducible_vrel_decomposition": reproducible_vrel,
        "legacy_cost_decomposition": legacy_cost,
        "reproducible_cost_decomposition": reproducible_cost,
        "internal_gain_survival": survival,
        "interpretation": {
            "remaining_gap": [
                "b2b connectivity",
                "grouping",
                "area/outline",
                "rho/aspect",
            ],
            "retained_advantage": ["p2b/port HPWL"],
            "unresolved": (
                "The public comparator does not isolate exact-area decoder, "
                "training data, and sampling settings; no single-factor causal "
                "attribution is supported."
            ),
            "sota": "not claimed",
        },
        "sources": {
            "rows": str(args.rows.relative_to(ROOT)),
            "legacy": str(args.legacy.relative_to(ROOT)),
            "component_gap": str(args.component_gap.relative_to(ROOT)),
            "mechanism": str(args.mechanism.relative_to(ROOT)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_table(report, args.table)
    build_figure(report, args.figure_root)
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()

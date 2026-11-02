#!/usr/bin/env python3
"""Build the FEVP locality-quality mechanism evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "typed_edit_graph",
    "dense_cross_attention",
    "no_edit_gating",
    "no_structural_graph",
)
METRICS = (
    "outside_pixel_exact",
    "outside_room_exact",
    "delta_iou",
    "delta_mse",
    "conditional_watertight",
    "node_f1",
    "edge_overlap",
    "ged",
)


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_locality_mechanism_v2.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "paper/figures/fp_locality_efficiency_tradeoff_v2.svg",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fp_locality_mechanism_v2.md",
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=ROOT / "docs/fp_locality_mechanism_claims_v1.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = read(
        "outputs/reports/fp_typed_edit_graph_ablation_rows_v1.json"
    )
    protocol = read("outputs/reports/fp_strict_locality_protocol_v2.json")
    rows = payload["rows"]
    summary = {
        arm: {
            metric: float(
                np.mean([float(row["arms"][arm][metric]) for row in rows])
            )
            for metric in METRICS
        }
        for arm in ARMS
    }
    failures = {}
    for arm in ARMS:
        pixel_failures = sum(
            float(row["arms"][arm]["outside_pixel_exact"]) < 1.0
            for row in rows
        )
        room_failures = sum(
            float(row["arms"][arm]["outside_room_exact"]) < 1.0
            for row in rows
        )
        contract_failures = sum(
            float(row["arms"][arm]["outside_pixel_exact"]) < 1.0
            or float(row["arms"][arm]["outside_room_exact"]) < 1.0
            for row in rows
        )
        failures[arm] = {
            "cases": len(rows),
            "outside_pixel_failures": pixel_failures,
            "outside_room_failures": room_failures,
            "contract_failures": contract_failures,
            "contract_failure_fraction": contract_failures / len(rows),
        }
    per_seed = defaultdict(dict)
    for row in rows:
        seed = str(row["seed"])
        for arm in ARMS:
            per_seed[seed].setdefault(arm, defaultdict(list))
            for metric in METRICS:
                per_seed[seed][arm][metric].append(
                    float(row["arms"][arm][metric])
                )
    per_seed_summary = {
        seed: {
            arm: {
                metric: float(np.mean(values))
                for metric, values in metrics.items()
            }
            for arm, metrics in arms.items()
        }
        for seed, arms in sorted(per_seed.items())
    }
    effects = {}
    for index, other in enumerate(ARMS[1:], start=1):
        key = f"typed_edit_graph_minus_{other}"
        effects[key] = {
            metric: paired_delta(
                [{"value": row["arms"]["typed_edit_graph"][metric]} for row in rows],
                [{"value": row["arms"][other][metric]} for row in rows],
                "value",
                iterations=5000,
                seed=12000 + index * 20 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    full = {
        "outside_pixel_exact": 1.0,
        "delta_iou": protocol["main_four_blocks"]["editing_quality"][
            "delta_iou"
        ]["mean"],
        "conditional_watertight": protocol["main_four_blocks"][
            "topology_legality"
        ]["conditional_watertightness"]["mean"],
        "node_f1": protocol["main_four_blocks"]["graph_consistency"][
            "node_f1"
        ]["mean"],
        "edge_overlap": protocol["main_four_blocks"]["graph_consistency"][
            "edge_overlap"
        ]["mean"],
    }
    arm_frontier = _frontier(summary)
    with_full = summary | {"floorflow_control_gated": full}
    full_frontier = _frontier(with_full)
    checks = {
        "all_arms_share_rows": all(
            set(row["arms"]) == set(ARMS) for row in rows
        ),
        "typed_preserves_outside_exact": (
            summary["typed_edit_graph"]["outside_pixel_exact"] == 1.0
            and summary["typed_edit_graph"]["outside_room_exact"] == 1.0
        ),
        "no_gate_violates_contract": (
            failures["no_edit_gating"]["contract_failures"] > 0
        ),
        "dense_attention_violates_contract": (
            failures["dense_cross_attention"]["contract_failures"] > 0
        ),
        "structural_graph_needed_for_quality": (
            summary["typed_edit_graph"]["delta_iou"]
            > summary["no_structural_graph"]["delta_iou"]
        ),
        "watertight_tradeoff_disclosed": (
            summary["typed_edit_graph"]["conditional_watertight"]
            < summary["no_structural_graph"]["conditional_watertight"]
        ),
        "full_control_gated_remains_frontier": (
            "floorflow_control_gated" in full_frontier
        ),
    }
    report = {
        "status": "complete",
        "scope": "five-seed pooled typed-routing rows",
        "cases": len(rows),
        "summary": summary,
        "contract_failures": failures,
        "per_seed": per_seed_summary,
        "paired_effects": effects,
        "floorflow_control_gated_reference": full,
        "frontiers": {
            "four_arm": arm_frontier,
            "including_full_control_gated": full_frontier,
        },
        "mechanism_claims": {
            "edit_gating_is_required": (
                "Removing edit gating causes strict-locality contract failures "
                "despite retaining nonzero edit response."
            ),
            "structural_routing_is_required_for_quality": (
                "With exact locality preserved, structural typed routing has "
                "higher Delta-IoU and better graph consistency than graphless routing."
            ),
            "dense_attention_is_not_a_protocol_substitute": (
                "Dense routing leaks outside the declared region and has lower "
                "edit quality than typed routing."
            ),
            "typed_routing_is_not_the_full_method": (
                "Typed routing is a mechanism ablation over rendered outputs; "
                "the full control-gated FloorFlow remains the quality-locality frontier."
            ),
        },
        "checks": checks,
        "sources": [
            "outputs/reports/fp_typed_edit_graph_ablation_rows_v1.json",
            "outputs/reports/fp_typed_edit_graph_ablation_v1.json",
            "outputs/reports/fp_strict_locality_protocol_v2.json",
        ],
    }
    failures_check = [name for name, passed in checks.items() if not passed]
    if failures_check:
        raise ValueError(f"locality mechanism checks failed: {failures_check}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_table(report, args.table)
    _write_claims(report, args.claims)
    _write_figure(report, args.figure)
    print(json.dumps({"status": report["status"], **checks}, indent=2))


def _frontier(points: dict[str, dict]) -> list[str]:
    output = []
    for name, point in points.items():
        dominated = any(
            other["outside_pixel_exact"] >= point["outside_pixel_exact"]
            and other["delta_iou"] >= point["delta_iou"]
            and (
                other["outside_pixel_exact"] > point["outside_pixel_exact"]
                or other["delta_iou"] > point["delta_iou"]
            )
            for other_name, other in points.items()
            if other_name != name
        )
        if not dominated:
            output.append(name)
    return output


def _write_table(report: dict, path: Path) -> None:
    lines = [
        "# Locality-Quality Mechanism v2",
        "",
        "| Arm | Outside exact | Contract failures | Delta-IoU | Watertight | Node F1 | Edge | GED |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in report["summary"].items():
        failure = report["contract_failures"][arm]
        lines.append(
            f"| {arm} | {values['outside_pixel_exact']:.6f} | "
            f"{failure['contract_failures']}/{failure['cases']} | "
            f"{values['delta_iou']:.6f} | "
            f"{values['conditional_watertight']:.6f} | "
            f"{values['node_f1']:.6f} | {values['edge_overlap']:.6f} | "
            f"{values['ged']:.6f} |"
        )
    full = report["floorflow_control_gated_reference"]
    lines.extend(
        [
            "",
            (
                f"Full control-gated FloorFlow reference: outside exact "
                f"{full['outside_pixel_exact']:.3f}, Delta-IoU "
                f"{full['delta_iou']:.6f}."
            ),
            "",
            (
                "The average outside-exact gap is categorical under FEVP: every "
                "case below 1.0 is a contract failure, not a small quality loss."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_claims(report: dict, path: Path) -> None:
    text = [
        "# Locality Mechanism Claims v1",
        "",
        "## Allowed",
        "",
        *[
            f"- {claim}"
            for claim in report["mechanism_claims"].values()
        ],
        "",
        "## Forbidden",
        "",
        "- Typed routing is the full FloorFlow method.",
        "- Typed routing improves watertightness.",
        "- A 0.99 outside average is acceptable under FEVP.",
        "- Dense attention is a valid strict-locality substitute.",
        "- The typed arm is a retrained cross-attention mechanism.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def _write_figure(report: dict, path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    points = report["summary"] | {
        "floorflow_control_gated": report["floorflow_control_gated_reference"]
    }
    labels = {
        "typed_edit_graph": "Typed routing",
        "dense_cross_attention": "Dense",
        "no_edit_gating": "No gate",
        "no_structural_graph": "No graph",
        "floorflow_control_gated": "Full FloorFlow",
    }
    colors = {
        "typed_edit_graph": "#16837a",
        "dense_cross_attention": "#2f6fb6",
        "no_edit_gating": "#b84a5a",
        "no_structural_graph": "#b77a00",
        "floorflow_control_gated": "#3f8f5f",
    }
    figure, axis = plt.subplots(figsize=(5.8, 3.4), constrained_layout=True)
    axis.axvspan(0.9855, 0.999999, color="#f4e7e9", alpha=0.75)
    axis.axvline(1.0, color="#526173", linestyle="--", linewidth=1.0)
    label_offsets = {
        "typed_edit_graph": (-88, 5),
        "dense_cross_attention": (-78, -12),
        "no_edit_gating": (8, 7),
        "no_structural_graph": (8, -14),
        "floorflow_control_gated": (8, 5),
    }
    for name, point in points.items():
        axis.scatter(
            point["outside_pixel_exact"],
            point["delta_iou"],
            s=70 if name in report["frontiers"]["four_arm"] else 45,
            color=colors[name],
            marker="*" if name == "floorflow_control_gated" else "o",
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )
        axis.annotate(
            labels[name],
            (point["outside_pixel_exact"], point["delta_iou"]),
            xytext=label_offsets[name],
            textcoords="offset points",
            fontsize=7,
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.5,
                "color": colors[name],
            },
        )
    axis.set_xlim(0.9855, 1.0005)
    axis.set_ylim(0.84, 1.0)
    axis.set_xlabel("Outside pixel exact (FEVP threshold = 1.0)")
    axis.set_ylabel("Delta-IoU")
    axis.set_title("Strict locality and quality are distinct mechanisms", loc="left")
    axis.text(
        0.986,
        0.985,
        "FEVP contract failure",
        ha="left",
        va="top",
        fontsize=7,
        color="#b84a5a",
    )
    axis.grid(True, color="#c7ced8", alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


__all__ = ["main"]


if __name__ == "__main__":
    main()

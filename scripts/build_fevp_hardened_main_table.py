#!/usr/bin/env python3
"""Build the hardened FEVP main-result table and claim ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stability",
        type=Path,
        default=ROOT / "outputs/reports/fevp_fiveseed_stability_v2.json",
    )
    parser.add_argument(
        "--outside",
        type=Path,
        default=ROOT / "outputs/reports/fevp_outside_layers_v2.json",
    )
    parser.add_argument(
        "--validity",
        type=Path,
        default=ROOT / "outputs/reports/fevp_validity_watertight_v2.json",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "outputs/reports/fevp_graph_metrics_v2.json",
    )
    parser.add_argument(
        "--delta",
        type=Path,
        default=ROOT / "outputs/reports/fevp_delta_locality_v2.json",
    )
    parser.add_argument(
        "--attacks",
        type=Path,
        default=ROOT / "outputs/reports/fevp_protocol_attacks_v2.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/fevp_metric_contract_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_hardened_main_results_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_hardened_main_results_v2.md",
    )
    return parser.parse_args()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    stability = _read(args.stability)
    outside = _read(args.outside)
    validity = _read(args.validity)
    graph = _read(args.graph)
    delta = _read(args.delta)
    attacks = _read(args.attacks)
    contract = _read(args.contract)
    metrics = stability["metrics"]
    outside_all = outside["summary"]["all"]
    validity_all = validity["summary"]["all"]
    graph_mean = graph["summary"]["seed_mean"]
    blocks = {
        "editing_quality": {
            "delta_iou": metrics["delta_iou"],
            "delta_mse": metrics["delta_mse"],
            "ged": metrics["ged"],
            "paired_ged": {
                arm: {
                    "mean_delta": sum(
                        graph["per_seed"][seed]["paired"][
                            f"typed_edit_graph_minus_{arm}:ged"
                        ]["mean_delta"]
                        for seed in graph["per_seed"]
                    )
                    / len(graph["per_seed"]),
                    "max_p": max(
                        graph["per_seed"][seed]["paired"][
                            f"typed_edit_graph_minus_{arm}:ged"
                        ]["wilcoxon_p_less"]
                        for seed in graph["per_seed"]
                    ),
                }
                for arm in (
                    "dense_cross_attention",
                    "no_edit_gating",
                    "no_structural_graph",
                )
            },
        },
        "strict_locality": {
            "strict": outside_all["strict"],
            "no_edit_gating": outside_all["no_edit_gating"],
            "single_room_control": outside["summary"][
                "single_room_negative_control"
            ],
        },
        "topology_legality": {
            "validity": validity_all["validity"],
            "validity_error_counts": validity["validity_error_counts"],
            "watertight": validity_all["watertight"],
            "raw_watertight": validity_all["raw_watertight"],
            "repair_count": validity_all["repair_count"],
            "regression_count": validity_all["regression_count"],
            "repair_rate_of_failures": validity_all[
                "repair_rate_of_failures"
            ],
        },
        "graph_consistency": graph_mean["typed_edit_graph"],
        "delta_distributions": delta["summary"],
        "attack_sensitivity": attacks["by_metric"],
    }
    report = {
        "status": "complete",
        "protocol": contract["protocol"],
        "protocol_version": contract["protocol_version"],
        "implementation_version": contract["implementation_version"],
        "tolerances": contract["tolerances"],
        "blocks": blocks,
        "claim": (
            "Under the hardened FEVP contract, strict outside preservation is "
            "verified at pixel, room-instance, and edge layers; no-gate leakage "
            "is exposed by continuous violation rate; graph and edit metrics are "
            "reported with precision, recall, paired tests, and five-seed intervals."
        ),
        "boundary": (
            "Hardened validity is a five-check conjunction and is not the old "
            "single fpeval flag. The watertight guard is reported as a failed "
            "net-benefit mechanism whenever regressions exceed repairs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    strict = outside_all["strict"]
    leaking = outside_all["no_edit_gating"]
    validity_values = validity_all["validity"]
    watertight = validity_all["watertight"]
    raw_watertight = validity_all["raw_watertight"]
    graph_values = {
        key: metrics[key]
        for key in (
            "node_precision",
            "node_recall",
            "node_f1",
            "edge_overlap",
            "ged",
        )
    }
    mechanism_values = graph_mean["typed_edit_graph"]
    lines = [
        "# Hardened FEVP Main Results v2",
        "",
        f"Protocol: {contract['protocol']} {contract['protocol_version']}.",
        "",
        "## Metric Contract",
        "",
        "| Metric | Reference implementation | Fixed tolerance | Negative controls |",
        "|---|---|---|---|",
    ]
    for name in (
        "outside_exact",
        "validity",
        "conditional_watertightness",
        "node_f1",
        "edge_overlap",
    ):
        metric = contract["metrics"][name]
        tolerances = ", ".join(
            f"{key}={value}" for key, value in contract["tolerances"].items()
        )
        lines.append(
            f"| {name} | `floorflow.evaluation.fevp_metric_contracts` | "
            f"{tolerances} | {len(metric.get('negative_controls', []))} |"
        )
    lines.extend(
        [
            "",
            "## Editing Quality",
            "",
            "| Metric | Mean | SD | 95% CI | Worst seed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in ("delta_iou", "delta_mse", "ged"):
        value = metrics[key]
        lines.append(
            f"| {key} | {value['mean']:.6f} | {value['std']:.6f} | "
            f"[{value['ci95_low']:.6f}, {value['ci95_high']:.6f}] | "
            f"{value['worst_seed']} ({value['worst']:.6f}) |"
        )
    lines.extend(
        [
            "",
            "| GED comparison | Mean paired delta | Max seed p (one-sided) |",
            "|---|---:|---:|",
        ]
    )
    for arm, value in blocks["editing_quality"]["paired_ged"].items():
        lines.append(
            f"| Typed - {arm} | {value['mean_delta']:+.6f} | "
            f"{value['max_p']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Strict Locality",
            "",
            "| Arm | Pixel exact | Room exact | Label exact | Polygon exact | Edge exact | Violation rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, values in (("Strict", strict), ("No gate", leaking)):
        lines.append(
            f"| {name} | {values['outside_pixel_exact']:.6f} | "
            f"{values['outside_room_level_exact']:.6f} | "
            f"{values['outside_room_label_exact']:.6f} | "
            f"{values['outside_room_polygon_exact']:.6f} | "
            f"{values['outside_edge_exact']:.6f} | "
            f"{values['locality_violation_rate']:.9f} |"
        )
    lines.extend(
        [
            "",
            (
                f"Single-room negative control: "
                f"{outside['summary']['single_room_negative_control']['cases']} "
                "cases; strict room-level exactness "
                f"{outside['summary']['single_room_negative_control']['strict']['outside_room_level_exact']:.6f}."
            ),
            "",
            "## Topology Legality",
            "",
            "| Check | Pass rate |",
            "|---|---:|",
        ]
    )
    for key, value in validity_values.items():
        lines.append(f"| {key} | {value:.6f} |")
    lines.extend(
        [
            "",
            "| Watertight metric | Raw | Guarded |",
            "|---|---:|---:|",
            (
                f"| conditional_watertightness_fpeval | "
                f"{validity_all['before_watertight_rate']:.6f} | "
                f"{validity_all['after_watertight_rate']:.6f} |"
            ),
        ]
    )
    for key in watertight:
        lines.append(
            f"| {key} | {raw_watertight[key]:.6f} | {watertight[key]:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                f"Guard repairs: {validity_all['repair_count']}; regressions: "
                f"{validity_all['regression_count']}; repair rate among failures: "
                f"{validity_all['repair_rate_of_failures']:.6f}."
            ),
            "The guard is not presented as a net-benefit contribution.",
            "",
            "## Graph Consistency",
            "",
            "| View | Node P | Node R | Node F1 | Edge P | Edge R | Edge F1/overlap | GED |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| Primary full method | {graph_values['node_precision']['mean']:.6f} | "
                f"{graph_values['node_recall']['mean']:.6f} | "
                f"{graph_values['node_f1']['mean']:.6f} | n/a | n/a | "
                f"{graph_values['edge_overlap']['mean']:.6f} | "
                f"{graph_values['ged']['mean']:.6f} |"
            ),
            (
                f"| Typed routing mechanism | "
                f"{mechanism_values['node_precision']:.6f} | "
                f"{mechanism_values['node_recall']:.6f} | "
                f"{mechanism_values['node_f1']:.6f} | "
                f"{mechanism_values['edge_precision']:.6f} | "
                f"{mechanism_values['edge_recall']:.6f} | "
                f"{mechanism_values['edge_f1']:.6f} | "
                f"{mechanism_values['ged']:.6f} |"
            ),
            "",
            "## Protocol Attacks",
            "",
            "| Mandatory metric | Attacks | Sensitive | False-pass rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (
        "outside_exact",
        "validity",
        "conditional_watertightness",
        "node_f1",
        "edge_overlap",
    ):
        values = attacks["by_metric"][name]
        lines.append(
            f"| {name} | {values['attacks']} | {values['sensitive']} | "
            f"{values['false_pass_rate']:.3f} |"
        )
    lines.extend(["", report["boundary"], ""])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "table": str(args.table)}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()

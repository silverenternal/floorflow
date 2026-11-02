"""Audit whether graph-protocol claims can be unified into one SOTA result."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    structural = json.loads(
        (
            ROOT / "outputs/reports/residential_structural_hardened_v2.json"
        ).read_text(encoding="utf-8")
    )
    typed = json.loads(
        (
            ROOT / "outputs/reports/fp_typed_edit_graph_ablation_v1.json"
        ).read_text(encoding="utf-8")
    )
    aligned = structural["protocols"]["house_mind_style_aligned"]
    reference = {
        "node_f1": 0.998,
        "edge_overlap": 0.934,
        "ged": 0.467,
    }
    ours = {
        metric: aligned[metric]["mean"]
        for metric in reference
    }
    comparisons = {
        arm.removeprefix("typed_edit_graph_minus_"): {
            metric: {
                "mean_delta": values[metric]["mean_delta"],
                "ci95": [
                    values[metric]["ci_low"],
                    values[metric]["ci_high"],
                ],
                "positive": values[metric]["ci_low"] > 0.0,
                "negative": values[metric]["ci_high"] < 0.0,
            }
            for metric in ("node_f1", "edge_overlap", "ged")
        }
        for arm, values in typed["effects"].items()
    }
    report = {
        "status": "blocked",
        "scope": "residential graph protocol dominance audit",
        "protocols": {
            "house_mind_style_aligned": {
                "ours": ours,
                "published_house_mind_e": reference,
                "wins": {
                    metric: (
                        ours[metric] > reference[metric]
                        if metric != "ged"
                        else ours[metric] < reference[metric]
                    )
                    for metric in reference
                },
            },
            "canonical_direct_tiled": {
                "metrics": structural["protocols"]["canonical_direct"][
                    "metrics"
                ],
                "external_comparable_baseline": False,
            },
            "typed_matched": {
                "metrics": structural["protocols"]["typed_matched"]["metrics"],
                "paired_effects": comparisons,
            },
        },
        "checks": {
            "aligned_beats_house_mind_on_all_three": all(
                report_win
                for report_win in (
                    ours["node_f1"] > reference["node_f1"],
                    ours["edge_overlap"] > reference["edge_overlap"],
                    ours["ged"] < reference["ged"],
                )
            ),
            "typed_beats_local_controls": all(
                all(
                    values["negative"]
                    if metric == "ged"
                    else values["positive"]
                    for metric, values in arm.items()
                )
                for arm in comparisons.values()
            ),
            "protocols_numerically_compatible": False,
        },
        "decision": (
            "All-protocol unified SOTA is not a valid claim. Aligned FloorFlow "
            "does not beat the published HouseMind-E values, canonical has no "
            "matched external baseline, and the typed result is an internal "
            "mechanism comparison with a different readout."
        ),
        "claim_boundary": (
            "Report a per-protocol dominance matrix. A win in one protocol "
            "cannot be transferred to another protocol or pooled into one score."
        ),
    }
    output = (
        ROOT / "outputs/reports/residential_graph_protocol_dominance_v1.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Graph Protocol Dominance",
        "",
        "| Protocol | Node F1 | Edge overlap | GED | Comparable baseline |",
        "|---|---:|---:|---:|---|",
        (
            f"| HouseMind-style aligned | {ours['node_f1']:.6f} | "
            f"{ours['edge_overlap']:.6f} | {ours['ged']:.6f} | "
            "HouseMind-E published anchor: 0.998 / 0.934 / 0.467 |"
        ),
        "| Canonical direct-tiled | 0.933189 | 0.889887 | 1.812344 | None |",
        "| Typed matched | 0.955068 | 0.837563 | 2.120855 | Local ablations only |",
        "",
        "No unified all-protocol SOTA claim is supported.",
    ]
    table = ROOT / "paper/tables/residential_graph_protocol_dominance_v1.md"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))


if __name__ == "__main__":
    main()

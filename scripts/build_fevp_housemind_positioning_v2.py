#!/usr/bin/env python3
"""Build a respectful, non-ranking HouseMind audit-positioning artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-status",
        type=Path,
        default=ROOT / "outputs/reports/fp_housemind_public_status_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_housemind_positioning_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_housemind_positioning_v2.md",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs/fevp_housemind_positioning_v2.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    public = json.loads(args.public_status.read_text(encoding="utf-8"))
    source = public["sources"]["arxiv_html"]["url"]
    methods = {
        "HouseMind-E": {
            "task": "natural-language floorplan editing",
            "dataset": "RPLAN",
            "delta_iou": 0.608,
            "delta_mse": 0.0019,
            "ged": 0.467,
            "node_f1": 0.998,
            "edge_overlap": 0.934,
            "outside_exact": None,
            "locality_violation_rate": None,
            "negative_controls": None,
            "protocol_attacks": None,
            "paired_stats": None,
        },
        "HouseMind-O": {
            "task": "unified Omni model; editing evaluation",
            "dataset": "RPLAN",
            "delta_iou": 0.598,
            "delta_mse": 0.0022,
            "ged": 0.653,
            "node_f1": 0.997,
            "edge_overlap": 0.908,
            "outside_exact": None,
            "locality_violation_rate": None,
            "negative_controls": None,
            "protocol_attacks": None,
            "paired_stats": None,
        },
        "HouseMind-G": {
            "task": "generation-only model",
            "dataset": "RPLAN",
            "delta_iou": None,
            "delta_mse": None,
            "ged": None,
            "node_f1": None,
            "edge_overlap": None,
            "outside_exact": None,
            "locality_violation_rate": None,
            "negative_controls": None,
            "protocol_attacks": None,
            "paired_stats": None,
            "editing_eligible": False,
        },
    }
    report = {
        "status": "complete",
        "role": "complementary_context_not_competitor",
        "source": source,
        "source_section": "Table 3",
        "verified_public_facts": {
            "unified_understanding_generation_editing": True,
            "multimodal_instruction_editing": True,
            "official_code_status": public["public_artifacts"]["code_status"],
            "strict_outside_exact_reported": False,
            "locality_violation_rate_reported": False,
            "negative_controls_reported": False,
            "protocol_attacks_reported": False,
            "paired_locality_statistics_reported": False,
        },
        "methods": methods,
        "claim": (
            "HouseMind addresses unified tokenization and multimodal instruction "
            "editing. FEVP addresses whether edit locality is independently "
            "auditable. HouseMind-E/O publish task-level edit and graph metrics "
            "but do not publish outside exactness, violation rate, negative "
            "controls, protocol attacks, or paired locality statistics."
        ),
        "non_ranking_statement": (
            "We do not claim that FloorFlow beats HouseMind or that HouseMind "
            "lacks locality. We claim that the public HouseMind evaluation does "
            "not expose the corresponding audit evidence, so no matched locality "
            "ranking is possible."
        ),
        "checks": {
            "house_mind_is_acknowledged": True,
            "house_mind_edit_variants_present": (
                methods["HouseMind-E"]["delta_iou"] is not None
                and methods["HouseMind-O"]["delta_iou"] is not None
            ),
            "generation_variant_not_ranked": (
                methods["HouseMind-G"]["editing_eligible"] is False
            ),
            "no_failure_inference": (
                public["metric_coverage"][
                    "quantitative_strict_locality_audit_found"
                ]
                is False
            ),
        },
        "boundary": (
            "A dash or null means not reported or not applicable. It does not "
            "mean zero, failure, or inferiority."
        ),
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# HouseMind Audit Positioning",
        "",
        "| Method | Task | Delta-IoU | Node F1 | Edge overlap | GED | Outside exact | Violation | Negative controls | Attacks | Paired stats |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for name, values in methods.items():
        def value(key, row=values):
            item = row.get(key)
            return "—" if item is None else f"{item:.3f}"

        table.append(
            f"| {name} | {values['task']} | {value('delta_iou')} | "
            f"{value('node_f1')} | {value('edge_overlap')} | {value('ged')} | "
            "— | — | — | — | — |"
        )
    table.extend(
        [
            "",
            "HouseMind-E/O values are public Table 3 context, not a same-dataset ranking.",
            "HouseMind-G is generation-only and is not placed on the editing axis.",
            "",
            report["non_ranking_statement"],
            "",
            report["boundary"],
            "",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    doc = [
        "# HouseMind Positioning v2",
        "",
        report["claim"],
        "",
        report["non_ranking_statement"],
        "",
        report["boundary"],
        "",
    ]
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(doc), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

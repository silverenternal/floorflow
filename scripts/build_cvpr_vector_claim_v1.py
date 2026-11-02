#!/usr/bin/env python3
"""Generate vector-route claim sections and the FEVP-2 results table."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/reports/vector_edit_results_v1.json"
SIGNIFICANCE = ROOT / "outputs/reports/vector_edit_significance_v1.json"
BASELINE = ROOT / "outputs/reports/vector_edit_baselines_v1.json"
FAILURE = ROOT / "outputs/reports/vector_edit_failure_tail_v1.json"
COMPLEXITY = ROOT / "outputs/reports/fp_complexity_strata_fiveseed_v1.json"
INFERENCE = (
    ROOT
    / "outputs/reports/vector_residual_inference_multiseed_v1.json"
)
OUTPUT = ROOT / "outputs/reports/cvpr_vector_claim_v1.json"
CLAIM_MD = ROOT / "paper/cvpr_claim_sections_v2.md"
TABLE = ROOT / "paper/tables/vector_edit_main_results_v1.md"


def _range(results: dict, metric: str) -> tuple[float, float, float]:
    values = [
        operation[metric]["mean"]
        for operation in results["by_operation"].values()
    ]
    return min(values), max(values), sum(values) / len(values)


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    significance = json.loads(SIGNIFICANCE.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    complexity = json.loads(COMPLEXITY.read_text(encoding="utf-8"))["strata"]
    inference = json.loads(INFERENCE.read_text(encoding="utf-8"))
    validity = _range(results, "validity")
    node_f1 = _range(results, "node_f1")
    edge = _range(results, "edge_overlap")
    ged = _range(results, "ged")
    multiplicity = significance["multiplicity"]
    maximum_adjusted_p = multiplicity["maximum_adjusted_p"]
    external_rows = [
        row
        for operation in baseline["operations"].values()
        for row in operation["rows"]
    ]
    unique_cases = len(
        {row["case_id"].split("/", 1)[1] for row in external_rows}
    )
    external = {
        "outside": sum(
            row["surgical"]["preservation"]["outside_room_level_exact"]
            for row in external_rows
        )
        / len(external_rows),
        "validity": sum(
            row["surgical"]["topology"]["validity"]["validity"]
            for row in external_rows
        )
        / len(external_rows),
        "node_f1": sum(
            row["surgical"]["graph"]["node_f1"] for row in external_rows
        )
        / len(external_rows),
        "edge_overlap": sum(
            row["surgical"]["graph"]["edge_overlap"] for row in external_rows
        )
        / len(external_rows),
    }
    complex_watertight = {
        "rooms_8_plus": complexity["room_count"]["8+"]["metrics"][
            "conditional_watertight"
        ],
        "both_interfaces": complexity["interface_complexity"]["both"]["metrics"][
            "conditional_watertight"
        ],
        "edit_area_medium": complexity["edit_area_fraction"]["medium"]["metrics"][
            "conditional_watertight"
        ],
        "edit_area_large": complexity["edit_area_fraction"]["large"]["metrics"][
            "conditional_watertight"
        ],
    }
    wall_reductions = [
        row["wall"]["mean_relative_reduction"]
        for row in inference["seed_aggregate"].values()
    ]
    opening_reductions = [
        row["opening"]["mean_relative_reduction"]
        for row in inference["seed_aggregate"].values()
    ]
    claim = (
        "Under the FEVP-2 vector edit protocol, FloorFlow provides a complete "
        f"audited result set on {results['cases']:,} five-seed rows over "
        f"{unique_cases:,} CubiCasa test503 edit cases: room-polygon, "
        "wall-support, opening-support, and edge "
        "outside exactness are 1.0 in every operation, with Locality Violation "
        f"Area 0. Under the same scorer, five-seed validity is "
        f"{validity[0]:.3f}-{validity[1]:.3f}, Node F1 is "
        f"{node_f1[0]:.3f}-{node_f1[1]:.3f}, and Edge overlap is "
        f"{edge[0]:.3f}-{edge[1]:.3f}. Conditional watertightness on complex "
        f"room-count, interface, and medium/large edit-area strata is "
        f"{min(complex_watertight.values()):.3f}-"
        f"{max(complex_watertight.values()):.3f}. Against a matched Raster2Seq "
        "outside-frozen reconstruction wrapper, FloorFlow's paired improvements "
        f"in outside exactness, validity, Node F1, and Edge overlap have "
        f"positive 95% bootstrap intervals in every seed and "
        f"Bonferroni-adjusted Wilcoxon p <= {maximum_adjusted_p:.2e} "
        f"across {multiplicity['comparison_count']} paired tests."
        " Inference-time residual refinement, restricted to affected wall and "
        "opening proposals, keeps room-label, polygon, wall, opening, and edge "
        "outside exactness at 1.0 with Locality Violation Area 0, and reduces "
        "wall MAE by "
        f"{min(wall_reductions):.2%}-{max(wall_reductions):.2%} and opening "
        f"center MAE by {min(opening_reductions):.2%}-"
        f"{max(opening_reductions):.2%} across operations and five seeds."
    )
    boundaries = [
        "Raster2Seq is a matched reconstruction wrapper, not an editing-aware method.",
        "The HouseMind and RenoLayout complete methods are not reproduced.",
        "CE2EPlan, FMLM, and TLC-Plan remain generation-only context.",
        "The result is specific to CubiCasa test503 and the FEVP-2 vector protocol.",
        "Strict locality excludes the declared edit region and changed wall support.",
        "No natural-language understanding or human-preference claim is made.",
    ]
    report = {
        "status": "complete",
        "claim": claim,
        "metadata": {
            "cases": results["cases"],
            "unique_cases": unique_cases,
            "seeds": results["seeds"],
            "validity_range": list(validity[:2]),
            "node_f1_range": list(node_f1[:2]),
            "edge_overlap_range": list(edge[:2]),
            "ged_range": list(ged[:2]),
            "paired_max_wilcoxon_p": multiplicity["maximum_raw_p"],
            "paired_max_bonferroni_p": maximum_adjusted_p,
            "paired_comparison_count": multiplicity["comparison_count"],
            "raster2seq_wrapper": external,
            "complex_conditional_watertight": complex_watertight,
            "inference_wall_relative_reduction_range": [
                min(wall_reductions),
                max(wall_reductions),
            ],
            "inference_opening_relative_reduction_range": [
                min(opening_reductions),
                max(opening_reductions),
            ],
            "inference_checkpoint": inference["checkpoint"],
            "inference_checkpoint_sha256": inference["checkpoint_sha256"],
            "inference_wilcoxon_max_bonferroni_p": inference["multiplicity"][
                "maximum_adjusted_p"
            ],
            "inference_comparison_count": inference["multiplicity"][
                "comparison_count"
            ],
            "failure_tail_locality": failure["checks"][
                "all_failure_tails_preserve_locality"
            ],
        },
        "boundaries": boundaries,
        "checks": {
            "strict_locality_all_operations": all(
                results["checks"].values()
            ),
            "paired_significance_all_seeds": all(
                significance["checks"].values()
            ),
            "failure_tails_preserve_locality": failure["checks"][
                "all_failure_tails_preserve_locality"
            ],
            "complex_watertight_over_0_96": min(
                complex_watertight.values()
            )
            >= 0.96,
            "inference_residual_complete": all(
                inference["checks"].values()
            ),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    CLAIM_MD.write_text(
        "\n".join(
            [
                "# CVPR Vector-Claim Sections v2",
                "",
                "This file contains canonical claim sections for internal review.",
                "It is not a complete manuscript.",
                "",
                "## Abstract Claim",
                "",
                claim,
                "",
                "## Boundary Statements",
                "",
                *[f"- {value}" for value in boundaries],
                "",
                "## Related-Work Positioning",
                "",
                (
                    "HouseMind and RenoLayout target unified multimodal reasoning "
                    "and renovation breadth. HouseMind uses discrete room-instance "
                    "tokens for understanding, generation, and editing. CE2EPlan "
                    "performs end-to-end boundary-conditioned vector floorplan "
                    "generation with topology- and geometry-enhanced components. "
                    "FMLM and TLC-Plan target unified or hierarchical vector "
                    "generation. Raster2Seq is a matched reconstruction baseline "
                    "and not an editing-aware model. CE2EPlan is a generation-only "
                    "target; no generation result is interpreted as an editing "
                    "result. The contribution "
                    "is protocol and evidence architecture. We do not claim that "
                    "FloorFlow beats HouseMind. Missing metrics define an audit "
                    "boundary; absence defines an audit boundary, not a "
                    "performance ranking."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        "# FEVP-2 Main Vector Results v1",
        "",
        (
            f"{results['cases']} matched rows; five-seed Student-t intervals are "
            "reported in the source JSON."
        ),
        "",
        "| Operation | Cases/seed | Validity | Node F1 | Edge overlap | GED |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for operation, values in results["by_operation"].items():
        lines.append(
            f"| {operation} | "
            f"{results['cases_per_operation_per_seed'][operation]} | "
            f"{values['validity']['mean']:.4f} | "
            f"{values['node_f1']['mean']:.4f} | "
            f"{values['edge_overlap']['mean']:.4f} | "
            f"{values['ged']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "All operations have outside room/wall/opening/edge exactness "
                "1.0 and Locality Violation Area 0."
            ),
        ]
    )
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "checks": report["checks"]}, indent=2))
    raise SystemExit(0 if all(report["checks"].values()) else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

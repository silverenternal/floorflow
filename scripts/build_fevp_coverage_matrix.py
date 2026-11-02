#!/usr/bin/env python3
"""Build the FEVP v1.0 protocol-coverage matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.fevp_protocol import (
    MANDATORY_METRICS,
    assert_valid_report,
)

ROOT = Path(__file__).resolve().parents[1]
METRIC_LABELS = {
    "outside_exact": "Outside exact",
    "validity": "Validity",
    "conditional_watertightness": "Watertight",
    "node_f1": "Node F1",
    "edge_overlap": "Edge overlap",
    "delta_iou": "Delta-IoU",
    "ged": "GED",
    "delta_mse": "Delta-MSE",
}


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_coverage_matrix_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_coverage_matrix_v1.md",
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=ROOT / "docs/fevp_claim_language_v1.md",
    )
    parser.add_argument(
        "--source-map",
        type=Path,
        default=ROOT / "outputs/reports/fevp_claim_source_map_v1.json",
    )
    return parser.parse_args()


def metric(status: str, *, value=None, source=None) -> dict:
    return {
        "status": status,
        "value": value,
        "source": source,
    }


def outside(
    status: str,
    *,
    pixel=None,
    room=None,
    source=None,
) -> dict:
    return {
        "status": status,
        "value": None,
        "outside_pixel_exact": pixel,
        "outside_room_label_exact": room,
        "source": source,
    }


def report(
    *,
    report_id: str,
    method: str,
    dataset: str,
    task: str,
    comparability: str,
    matched_group: str | None,
    mandatory: dict,
    optional: dict,
    cases: int,
    source_map: dict,
    notes: str,
) -> dict:
    return {
        "schema_version": "fevp-report-v1",
        "protocol_version": "1.0",
        "report_id": report_id,
        "method": method,
        "dataset": dataset,
        "task": task,
        "comparability": {
            "status": comparability,
            "matched_group": matched_group,
            "notes": notes,
        },
        "edit_region": {
            "definition": (
                "Declared local edit rectangle for FloorFlow; external methods "
                "do not expose a matched FEVP rectangle."
            ),
            "rasterization_rule": (
                "Clamp and round normalized coordinates; not applicable where "
                "external edit masks are absent."
            ),
            "valid_outside_rule": (
                "Footprint outside the edit region, excluding dilated pre/post walls."
            ),
        },
        "scorer": {
            "implementation": "floorflow.evaluation.fevp_protocol",
            "version": "1.0",
            "code_commit": "working-tree",
        },
        "mandatory": mandatory,
        "optional": optional,
        "cases": {
            "count": cases,
            "rows_file": None,
            "sha256": None,
        },
        "source_map": source_map,
        "excluded_components": ["human_evaluation"],
    }


def main() -> None:
    args = parse_args()
    protocol = read("outputs/reports/fp_strict_locality_protocol_v2.json")
    house = read("outputs/reports/fp_housemind_public_status_v2.json")
    pareto = read("outputs/reports/fp_pareto_matched_v1.json")
    raster2seq_official = read(
        "outputs/reports/fp_raster2seq_official_anchor_v1.json"
    )
    ce2eplan = read("outputs/reports/ce2eplan_official_anchor_v1.json")
    hardened = read("outputs/reports/fevp_hardened_main_results_v2.json")
    blocks = protocol["main_four_blocks"]
    floorplan_sources = {
        "outside_exact": (
            "outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json"
        ),
        "validity": "outputs/reports/fevp_hardened_main_results_v2.json",
        "conditional_watertightness": (
            "outputs/reports/fp_verifiability_score_v1.json"
        ),
        "node_f1": "outputs/reports/fp_housemind_structural_alignment_v1.json",
        "edge_overlap": (
            "outputs/reports/fp_housemind_structural_alignment_v1.json"
        ),
        "delta_iou": (
            "outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json"
        ),
        "ged": "outputs/reports/fp_housemind_structural_alignment_v1.json",
        "delta_mse": (
            "outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json"
        ),
    }
    reports = {}
    reports["FloorFlow"] = report(
        report_id="floorflow-fevp-v1",
        method="FloorFlow control-gated local delta",
        dataset="CubiCasa5K test503",
        task="declared local editing",
        comparability="matched",
        matched_group="cubicasa-test503-five-seed",
        mandatory={
            "outside_exact": outside(
                "reported_under_fevp",
                pixel=1.0,
                room=1.0,
                source=floorplan_sources["outside_exact"],
            ),
            "validity": metric(
                "reported_under_fevp",
                value=hardened["blocks"]["topology_legality"]["validity"][
                    "validity"
                ],
                source=floorplan_sources["validity"],
            ),
            "conditional_watertightness": metric(
                "reported_under_fevp",
                value=blocks["topology_legality"][
                    "conditional_watertightness"
                ]["mean"],
                source=floorplan_sources["conditional_watertightness"],
            ),
            "node_f1": metric(
                "reported_under_fevp",
                value=blocks["graph_consistency"]["node_f1"]["mean"],
                source=floorplan_sources["node_f1"],
            ),
            "edge_overlap": metric(
                "reported_under_fevp",
                value=blocks["graph_consistency"]["edge_overlap"]["mean"],
                source=floorplan_sources["edge_overlap"],
            ),
        },
        optional={
            "delta_iou": metric(
                "reported_under_fevp",
                value=blocks["editing_quality"]["delta_iou"]["mean"],
                source=floorplan_sources["delta_iou"],
            ),
            "ged": metric(
                "reported_under_fevp",
                value=blocks["editing_quality"]["ged"]["mean"],
                source=floorplan_sources["ged"],
            ),
            "delta_mse": metric(
                "reported_under_fevp",
                value=blocks["editing_quality"]["delta_mse"]["mean"],
                source=floorplan_sources["delta_mse"],
            ),
        },
        cases=12395,
        source_map=floorplan_sources,
        notes=(
            "Complete hardened FEVP row; validity is the five-check conjunction "
            "and structural metrics use oracle controls."
        ),
    )
    house_source = house["sources"]["arxiv_html"]["url"]
    for method, task, node, edge, delta, ged, mse in (
        (
            "HouseMind-E",
            "natural-language floorplan editing",
            0.998,
            0.934,
            0.608,
            0.467,
            0.0019,
        ),
        (
            "HouseMind-O",
            "unified Omni model editing evaluation",
            0.997,
            0.908,
            0.598,
            0.653,
            0.0022,
        ),
    ):
        reports[method] = report(
            report_id=f"{method.lower()}-public-context",
            method=method,
            dataset="RPLAN natural-language editing",
            task=task,
            comparability="not_comparable",
            matched_group=None,
            mandatory={
                "outside_exact": outside("not_comparable"),
                "validity": metric("not_comparable"),
                "conditional_watertightness": metric("not_comparable"),
                "node_f1": metric("reported_outside_fevp", value=node, source=house_source),
                "edge_overlap": metric("reported_outside_fevp", value=edge, source=house_source),
            },
            optional={
                "delta_iou": metric("reported_outside_fevp", value=delta, source=house_source),
                "ged": metric("reported_outside_fevp", value=ged, source=house_source),
                "delta_mse": metric("reported_outside_fevp", value=mse, source=house_source),
            },
            cases=2308,
            source_map={key: house_source for key in ("node_f1", "edge_overlap", "delta_iou", "ged", "delta_mse")},
            notes="Public editing context; no FEVP locality-audit evidence.",
        )
    reports["HouseMind-G"] = report(
        report_id="housemind-g-generation-context",
        method="HouseMind-G",
        dataset="RPLAN generation",
        task="generation-only; not an editing baseline",
        comparability="not_comparable",
        matched_group=None,
        mandatory={
            name: metric("not_comparable") for name in MANDATORY_METRICS
        },
        optional={
            name: metric("not_comparable")
            for name in ("delta_iou", "ged", "delta_mse")
        },
        cases=0,
        source_map={},
        notes=(
            "Generation-only model. It is intentionally excluded from the "
            "editing Pareto and all editing claims."
        ),
    )
    methods = {
        row["method"]: row
        for row in pareto["matched_methods"]
    }
    for label, key, kind in (
        ("Raster2Seq native", "Raster2Seq native", "reconstruction"),
        (
            "Raster2Seq outside-frozen wrapper",
            "Raster2Seq outside-frozen redraw",
            "editing wrapper",
        ),
    ):
        row = methods[key]
        source = "outputs/reports/fp_pareto_matched_v1.json"
        reports[label] = report(
            report_id=f"{label.lower().replace(' ', '-')}-context",
            method=key,
            dataset="CubiCasa5K test503",
            task=kind,
            comparability="matched",
            matched_group="cubicasa-test503-edit-rows",
            mandatory={
                "outside_exact": outside(
                    "reported_outside_fevp",
                    pixel=row["outside_pixel_exact"],
                    room=None,
                    source=source,
                ),
                "validity": metric("not_reported"),
                "conditional_watertightness": metric("not_reported"),
                "node_f1": metric("not_reported"),
                "edge_overlap": metric("not_reported"),
            },
            optional={
                "delta_iou": metric(
                    "reported_outside_fevp",
                    value=row["delta_iou"],
                    source=source,
                ),
                "ged": metric("not_reported"),
                "delta_mse": metric(
                    "reported_outside_fevp",
                    value=row["delta_mse"],
                    source=source,
                ),
            },
            cases=2479,
            source_map={
                "outside_exact": source,
                "delta_iou": source,
                "delta_mse": source,
            },
            notes="Reconstruction or outside-frozen wrapper, not native editing.",
        )
    reports["Raster2Seq official anchor"] = report(
        report_id="raster2seq-official-reconstruction-anchor",
        method="Raster2Seq official evaluator",
        dataset="CubiCasa5K extracted test509",
        task="floorplan reconstruction",
        comparability="not_comparable",
        matched_group=None,
        mandatory={
            name: metric("not_comparable") for name in MANDATORY_METRICS
        },
        optional={
            name: metric("not_comparable")
            for name in ("delta_iou", "ged", "delta_mse")
        },
        cases=raster2seq_official["case_unit_audit"][
            "official_annotation_cases"
        ],
        source_map={},
        notes=(
            "Official RoomF1 0.8853 reproduces the public 0.887 anchor. "
            "Its contour-region cases cannot be bijected to FloorFlow's 503 "
            "SVG-floor cases; no matched identity row is claimed."
        ),
    )
    reports["CE2EPlan"] = report(
        report_id="ce2eplan-public-context",
        method="CE2EPlan",
        dataset="RPLAN boundary-conditioned generation",
        task="boundary-conditioned generation",
        comparability="not_comparable",
        matched_group=None,
        mandatory={
            name: metric("not_comparable") for name in MANDATORY_METRICS
        },
        optional={
            name: metric("not_comparable")
            for name in ("delta_iou", "ged", "delta_mse")
        },
        cases=ce2eplan["cases"],
        source_map={},
        notes=(
            "Official RPLAN anchor reproduced with FID "
            f"{ce2eplan['reproduced']['FID']:.4f} and independent GED "
            f"{ce2eplan['reproduced']['GED_mean']:.4f}. No license file exists, "
            "and no matched FloorFlow-on-RPLAN run exists."
        ),
    )
    for method_report in reports.values():
        assert_valid_report(method_report)
    matrix = {
        "schema_version": "fevp-coverage-matrix-v1",
        "protocol_version": "1.0",
        "status": "complete",
        "methods": reports,
        "claims": {
            "floorflow_complete_under_fevp": True,
            "external_unreported_cells_are_not_failures": True,
            "matched_floorflow_and_raster2seq": True,
            "raster2seq_official_anchor_reproduced": True,
            "raster2seq_503_matched_identity": False,
            "ce2eplan_official_anchor_reproduced": True,
            "ce2eplan_same_data_comparison": False,
            "human_evaluation_excluded": True,
            "hardened_metric_contract": True,
            "hardened_validity_is_five_check_conjunction": True,
        },
        "dash_definition": (
            "A dash means not reported under FEVP or not comparable; it does not "
            "mean zero, failure, or inferior performance."
        ),
        "sources": {
            "protocol": "outputs/reports/fp_strict_locality_protocol_v2.json",
            "house_mind": "outputs/reports/fevp_housemind_positioning_v2.json",
            "raster2seq": "outputs/reports/fp_pareto_matched_v1.json",
            "raster2seq_official": (
                "outputs/reports/fp_raster2seq_official_anchor_v1.json"
            ),
            "ce2eplan": "outputs/reports/ce2eplan_official_anchor_v1.json",
            "hardened_main": "outputs/reports/fevp_hardened_main_results_v2.json",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    _write_source_map(matrix, args.source_map)
    _write_table(matrix, args.table)
    _write_claims(matrix, args.claims)
    print(
        json.dumps(
            {
                "status": matrix["status"],
                "methods": list(reports),
                "floorflow_complete": True,
            },
            indent=2,
        )
    )


def _cell(metric_value: dict, *, outside_exact: bool = False) -> str:
    status = metric_value["status"]
    if status not in ("reported_under_fevp", "reported_outside_fevp"):
        return "—"
    if outside_exact:
        pixel = metric_value.get("outside_pixel_exact")
        room = metric_value.get("outside_room_label_exact")
        if pixel is None:
            return "partial"
        if room is None:
            return f"{pixel:.3f} px"
        return f"{pixel:.3f}/{room:.3f}"
    value = metric_value.get("value")
    return "—" if value is None else f"{value:.3f}"


def _write_table(matrix: dict, path: Path) -> None:
    lines = [
        "# FEVP v1.0 Protocol Coverage Matrix",
        "",
        "A dash means not reported under FEVP or not comparable; it does not mean failure.",
        "",
        "| Method | Outside exact | Validity | Watertight | Node F1 | Edge overlap | Delta-IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report_value in matrix["methods"].items():
        mandatory = report_value["mandatory"]
        optional = report_value["optional"]
        lines.append(
            f"| {name} | "
            f"{_cell(optional.get('outside_exact', mandatory['outside_exact']), outside_exact=True)} | "
            f"{_cell(mandatory['validity'])} | "
            f"{_cell(mandatory['conditional_watertightness'])} | "
            f"{_cell(mandatory['node_f1'])} | "
            f"{_cell(mandatory['edge_overlap'])} | "
            f"{_cell(optional['delta_iou'])} |"
        )
    lines.extend(
        [
            "",
            (
                "FloorFlow is the only complete evaluated FEVP row. The matrix "
                "does not rank methods on missing values."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_claims(matrix: dict, path: Path) -> None:
    lines = [
        "# FEVP Claim Language v1",
        "",
        "## Allowed",
        "",
        (
            "> Under FEVP v1.0, FloorFlow is the only method in the evaluated "
            "coverage matrix with a complete report for all five mandatory metrics."
        ),
        "",
        (
            "> HouseMind provides a complementary unified MLLM interface; its "
            "public paper/demo does not expose a matched FEVP strict-locality audit."
        ),
        "",
        (
            "> Raster2Seq provides a matched reconstruction reference; missing "
            "editing and FEVP fields are shown as N/A."
        ),
        "",
        (
            "> Raster2Seq's official RoomF1 and CE2EPlan's RPLAN FID provide "
            "external reproducibility anchors, but neither enters a matched "
            "editing leaderboard."
        ),
        "",
        "## Required N/A definition",
        "",
        f"> {matrix['dash_definition']}",
        "",
        "## Forbidden",
        "",
        "- HouseMind fails strict locality.",
        "- HouseMind or CE2EPlan is worse because a FEVP cell is missing.",
        "- The matrix is a cross-dataset leaderboard.",
        "- FloorFlow is the first strict-locality system globally.",
        "- CE2EPlan is a matched editing baseline.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_source_map(matrix: dict, path: Path) -> None:
    source_map = {
        method: report_value["source_map"]
        for method, report_value in matrix["methods"].items()
    }
    payload = {
        "status": "complete",
        "protocol_version": "1.0",
        "sources": source_map,
        "floorflow_required_metrics": {
            name: source_map["FloorFlow"].get(name)
            for name in MANDATORY_METRICS
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["main"]
if __name__ == "__main__":
    main()

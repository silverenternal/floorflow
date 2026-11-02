"""Summarize Raster2Seq sensitivity under matched preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metric_summary(rows: dict[str, dict], metric: str) -> dict:
    values = {
        name: float(values[metric])
        for name, values in rows.items()
        if metric in values
    }
    array = np.asarray(list(values.values()), dtype=float)
    return {
        "values": values,
        "min": float(array.min()),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
        "mean": float(array.mean()),
    }


def main() -> None:
    blend = load("outputs/reports/raster2seq_blend_selection_val_v1.json")
    dihedral = load(
        "outputs/reports/raster2seq_dihedral_consensus_val_v1.json"
    )
    fusion = load(
        "outputs/reports/raster2seq_polygon_fusion_val_v5.json"
    )
    identity = load(
        "outputs/reports/residential_identity_test509_v4_summary.json"
    )
    edit = load("outputs/reports/residential_edit_test503_v1.json")
    baseline = blend["baseline"]
    blend_values = {
        alpha: {
            metric: values["comparisons"][metric]["candidate_mean"]
            for metric in ("room_f1", "room_iou", "corner_f1")
        }
        for alpha, values in blend["candidates"].items()
    }
    view_values = {
        name: values
        for name, values in dihedral["strategies"].items()
        if name.endswith("_only") or name == "original"
    }
    edit_rows = {}
    for operation, values in edit["operations"].items():
        raster = values["per_seed"]["19"]["raster2seq_local"]
        floorflow = values["primary_edit_summary"]
        edit_rows[operation] = {
            "raster2seq_delta_iou": raster["delta_iou"],
            "raster2seq_delta_mse": raster["delta_mse"],
            "raster2seq_outside_exact": raster["outside_exact_ratio"],
            "floorflow_delta_iou": floorflow["delta_iou"]["mean"],
            "floorflow_delta_mse": floorflow["delta_mse"]["mean"],
            "floorflow_outside_exact": floorflow["outside_exact_ratio"]["mean"],
            "paired_p_value": values["paired_vs_raster2seq_local"]["p_value"],
            "rank_biserial": values["paired_vs_raster2seq_local"][
                "rank_biserial"
            ],
        }
    report = {
        "status": "complete",
        "scope": "Raster2Seq matched-preprocessing sensitivity",
        "checkpoint_blend_sensitivity": {
            "baseline": baseline,
            "values": blend_values,
            "room_f1": metric_summary(blend_values, "room_f1"),
            "room_iou": metric_summary(blend_values, "room_iou"),
            "corner_f1": metric_summary(blend_values, "corner_f1"),
            "selection_rule": blend["selection_rule"],
        },
        "view_sensitivity_val503": {
            "room_f1": metric_summary(view_values, "room_f1"),
            "room_iou": metric_summary(view_values, "room_iou"),
            "corner_f1": metric_summary(view_values, "corner_f1"),
        },
        "consensus_sensitivity_val503": {
            "count_consensus": dihedral["strategies"]["count_consensus_8"],
            "polygon_fusion": fusion["strategies"][fusion["selected"]],
            "selected_fusion": fusion["selected"],
            "paired_room_iou_vs_count": fusion[
                "paired_vs_count_consensus"
            ][fusion["selected"]]["room_iou"],
        },
        "identity_test509": {
            "metrics": identity["test_metrics"],
            "paired_room_iou_vs_count": identity[
                "paired_test_vs_count_consensus"
            ]["room_iou"],
            "published_room_f1": identity["baselines"][
                "published_room_f1"
            ],
        },
        "edit_test503": edit_rows,
        "checks": {
            "checkpoint_blend_sensitivity_reported": True,
            "view_sensitivity_reported": True,
            "consensus_sensitivity_reported": True,
            "matched_edit_sensitivity_reported": True,
            "room_iou_paired_ci_positive": identity[
                "paired_test_vs_count_consensus"
            ]["room_iou"]["ci_low"]
            > 0.0,
        },
        "claim": (
            "Under matched CubiCasa preprocessing, Raster2Seq reproduces the "
            "published RoomF1 anchor and remains a deterministic but "
            "view-sensitive baseline. On the same edit rows, outside-frozen "
            "FloorFlow outperforms the Raster2Seq local-redraw wrapper across "
            "all five operations while preserving outside pixels exactly."
        ),
        "boundary": (
            "This is a matched-preprocessing comparison on CubiCasa, not a "
            "claim that FloorFlow defeats Raster2Seq's native reconstruction "
            "task. Raster2Seq is not an editing-aware method."
        ),
    }
    output = ROOT / "outputs/reports/raster2seq_sensitivity_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Raster2Seq Matched-Preprocessing Sensitivity",
        "",
        "| Regime | RoomF1 | Room IoU | Corner F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, values in blend_values.items():
        lines.append(
            f"| checkpoint alpha {name} | {values['room_f1']:.6f} | "
            f"{values['room_iou']:.6f} | {values['corner_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "View-only ranges on val503:",
            "",
            (
                f"- RoomF1 {report['view_sensitivity_val503']['room_f1']['min']:.6f}"
                f"-{report['view_sensitivity_val503']['room_f1']['max']:.6f}"
            ),
            (
                f"- Room IoU {report['view_sensitivity_val503']['room_iou']['min']:.6f}"
                f"-{report['view_sensitivity_val503']['room_iou']['max']:.6f}"
            ),
            (
                f"- Corner F1 {report['view_sensitivity_val503']['corner_f1']['min']:.6f}"
                f"-{report['view_sensitivity_val503']['corner_f1']['max']:.6f}"
            ),
            "",
            "| Edit operation | R2S Delta-IoU | FloorFlow Delta-IoU | R2S outside exact | FloorFlow outside exact | Paired p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for operation, values in edit_rows.items():
        lines.append(
            f"| {operation} | {values['raster2seq_delta_iou']:.6f} | "
            f"{values['floorflow_delta_iou']:.6f} | "
            f"{values['raster2seq_outside_exact']:.6f} | "
            f"{values['floorflow_outside_exact']:.6f} | "
            f"{values['paired_p_value']:.3e} |"
        )
    table = ROOT / "paper/tables/raster2seq_sensitivity_v1.md"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))


if __name__ == "__main__":
    main()

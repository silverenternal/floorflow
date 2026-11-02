"""Build a machine-readable map of predicted-control SOTA blockers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]


def _mean(rows: list[dict], key: str) -> float:
    return fmean(float(row[key]) for row in rows)


def _selected_screen(run: Path) -> tuple[int, dict, list[dict]]:
    history = [
        json.loads(line)
        for line in (run / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    selected = max(history, key=lambda row: row["selection_score"])
    return int(selected["epoch"]), selected, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gaps",
        type=Path,
        default=ROOT / "outputs/reports/predicted_control_metric_gap_audit_v1.json",
    )
    parser.add_argument(
        "--full-val",
        type=Path,
        default=ROOT
        / "outputs/reports/predicted_control_full_stage_val491_gedfixed_v2.json",
    )
    parser.add_argument(
        "--screen",
        type=Path,
        default=ROOT
        / "outputs/train/typed_channel_v6_control_graph_adapter_screen256_seed19_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/predicted_sota_blocker_map_v1.json",
    )
    args = parser.parse_args()
    gap_report = json.loads(args.gaps.read_text(encoding="utf-8"))
    full_rows = json.loads(args.full_val.read_text(encoding="utf-8"))["cases"]
    selected_epoch, selected, history = _selected_screen(args.screen)
    gates = gap_report["gates"]
    blockers = []
    for metric in (
        "room_iou",
        "instance_f1",
        "wall_f1",
        "opening_f1",
        "node_f1",
        "edge_overlap",
        "ged",
    ):
        gate = gates[metric]
        blockers.append(
            {
                "metric": metric,
                "current": gate["value"],
                "gate": gate["gate"],
                "gap": gate["gap"],
                "higher_is_better": metric != "ged",
                "severity": (
                    "blocking"
                    if not gate["gate_passed"]
                    else "passing"
                ),
            }
        )
    diagnostics = {
        "wall_precision": _mean(full_rows, "fpeval_wall_precision"),
        "wall_recall": _mean(full_rows, "fpeval_wall_recall"),
        "instance_type_accuracy": _mean(
            full_rows,
            "instance_matched_type_accuracy",
        ),
        "instance_count_abs_error": _mean(
            full_rows,
            "instance_count_abs_error",
        ),
        "room_support_iou": _mean(full_rows, "room_support_iou"),
        "opening_vector_proxy_same_case": 0.0,
        "opening_pixel_proxy_same_case": 0.2048857368006304,
    }
    graph_screen = {
        "run": str(args.screen),
        "epochs_completed": len(history),
        "selected_epoch": selected_epoch,
        "selected": {
            name: float(selected[name])
            for name in (
                "room_macro_iou",
                "room_precision",
                "room_recall",
                "instance_f1",
                "instance_type_accuracy",
                "instance_count_abs_error",
                "node_f1",
                "edge_overlap",
                "ged",
            )
        },
        "first_epoch": {
            name: float(history[0][name])
            for name in (
                "room_macro_iou",
                "room_precision",
                "room_recall",
                "instance_f1",
                "instance_type_accuracy",
                "instance_count_abs_error",
                "node_f1",
                "edge_overlap",
                "ged",
            )
        },
    }
    room_precision_values = [float(row["room_precision"]) for row in history]
    room_recall_values = [float(row["room_recall"]) for row in history]
    opening_vector_report = json.loads(
        (
            ROOT
            / "outputs/train/typed_channel_v6_control_cpu_opening_vector_proxy_v2/report.json"
        ).read_text(encoding="utf-8")
    )["history"][-1]
    graph_warmstart_report = json.loads(
        (
            ROOT
            / "outputs/train/typed_channel_v6_room_graph_adapter_cpu_contract_v12/report.json"
        ).read_text(encoding="utf-8")
    )["history"][-1]
    report = {
        "schema_version": "predicted-control-sota-blocker-map-v1",
        "scope": "residential predicted-control only",
        "authoritative_full_val": str(args.full_val),
        "blockers": blockers,
        "diagnostics": diagnostics,
        "latest_graph_only_screen": graph_screen,
        "mechanism_diagnostics": {
            "dead_graph_output_control": (
                len(set(room_precision_values)) == 1
                and len(set(room_recall_values)) == 1
            ),
            "opening_pixel_f1": float(opening_vector_report["opening_f1"]),
            "opening_vector_f1": float(
                opening_vector_report["opening_vector_f1_proxy"]
            ),
            "graph_first_class_residual_l2": float(
                graph_warmstart_report[
                    "room_graph_first_class_residual_l2"
                ]
            ),
            "graph_second_class_residual_l2": float(
                graph_warmstart_report[
                    "room_graph_second_class_residual_l2"
                ]
            ),
        },
        "best_independent_frontier_not_composed": {
            "wall_f1_gap002": 0.5007057849606266,
            "opening_f1_centerwidth": 0.337977,
            "composition_status": "not_jointly_promoted",
            "remaining_gate_gaps": {
                "wall_f1": 0.0792942150393734,
                "opening_f1": 0.012023,
            },
        },
        "module_ownership": {
            "room_iou_instance_f1": [
                "room instance decoder",
                "room query set",
                "owner-preserving readout",
            ],
            "node_edge_ged": [
                "room identity",
                "relational adjacency",
                "count-selected graph conditioner",
                "canonical raster graph",
            ],
            "wall_f1": [
                "wall stage refiner",
                "room-boundary handoff",
                "wall candidate precision",
            ],
            "opening_f1": [
                "opening stage foreground/type",
                "wall host condition",
                "center-width vector decoder",
            ],
        },
        "primary_conclusion": (
            "The model cannot claim predicted-control SOTA because Room/Instance, "
            "Wall, Opening, Node/Edge/GED still fail absolute gates. Opening pixel "
            "metrics do not transfer to the vector consumer, and the completed "
            "graph-only control is invalid as a graph rejection because legacy "
            "loading restored zero graph-to-output heads."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

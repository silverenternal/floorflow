"""Summarize loss-to-metric failures in matched typed-channel training logs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _history(run: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def _summary(rows: list[dict]) -> dict:
    selected = max(rows, key=lambda row: row["selection_score"])
    edge_losses = [
        float(
            row["loss_components"].get(
                "dense_type_edge_loss",
                row["loss_components"].get("dense_type_adjacency"),
            )
        )
        for row in rows
    ]
    edge_mean = statistics.fmean(edge_losses)
    edge_spread = (
        max(edge_losses) - min(edge_losses)
    ) / max(abs(edge_mean), 1.0e-8)
    return {
        "epochs": len(rows),
        "selected_epoch": int(selected["epoch"]),
        "selected": {
            name: float(selected[name])
            for name in (
                "selection_score",
                "room_macro_iou",
                "instance_f1",
                "instance_count_abs_error",
                "node_f1",
                "edge_overlap",
                "ged",
            )
        },
        "final": {
            name: float(rows[-1][name])
            for name in (
                "selection_score",
                "room_macro_iou",
                "instance_f1",
                "instance_count_abs_error",
                "node_f1",
                "edge_overlap",
                "ged",
            )
        },
        "dense_edge_loss": {
            "mean": edge_mean,
            "min": min(edge_losses),
            "max": max(edge_losses),
            "relative_spread": edge_spread,
        },
        "graph_supervision": {
            "mean_valid_pairs": statistics.fmean(
                float(row["loss_components"]["room_adjacency_valid_pairs"])
                for row in rows
            ),
            "min_valid_pairs": min(
                float(row["loss_components"]["room_adjacency_valid_pairs"])
                for row in rows
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = _summary(_history(args.control))
    candidate = _summary(_history(args.candidate))
    report = {
        "schema_version": "typed-channel-screen-diagnostics-v1",
        "control": control,
        "candidate": candidate,
        "diagnoses": {
            "edge_objective_stagnant": (
                control["dense_edge_loss"]["relative_spread"] < 0.02
                and control["selected"]["edge_overlap"] < 0.5
            ),
            "full_model_prior_degraded": (
                control["selected_epoch"] == 0
                and control["final"]["instance_f1"]
                < control["selected"]["instance_f1"]
                and control["final"]["instance_count_abs_error"]
                > control["selected"]["instance_count_abs_error"]
            ),
            "graph_supervision_vanished": (
                min(
                    control["graph_supervision"]["min_valid_pairs"],
                    candidate["graph_supervision"]["min_valid_pairs"],
                )
                < 0.1
            ),
        },
        "claim_boundary": (
            "Diagnostic analysis of a single-seed mechanism screen; it does "
            "not establish model quality or SOTA."
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

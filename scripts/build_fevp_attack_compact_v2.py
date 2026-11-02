#!/usr/bin/env python3
"""Build the compact headline table for FEVP protocol attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "outputs/reports/fevp_protocol_attack_rows_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_protocol_attacks_compact_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_protocol_attacks_compact_v2.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.rows.read_text(encoding="utf-8"))["rows"]
    selected = []
    specs = (
        (
            "move_untouched_room_1px",
            "outside_room_level_exact",
            "Outside room-level exactness decreases",
        ),
        (
            "inject_one_pixel_hole",
            "hole_count",
            "Watertight hole count increases",
        ),
        (
            "delete_adjacency_edge_f1",
            "f1",
            "Edge F1 decreases",
        ),
        (
            "relabel_one_node_type_f1",
            "f1",
            "Node F1 decreases",
        ),
        (
            "overlap_two_rooms",
            "no_room_overlap",
            "Validity room-overlap check decreases",
        ),
    )
    for attack, submetric, expected in specs:
        row = next(
            item
            for item in rows
            if item["attack"] == attack and item["submetric"] == submetric
        )
        selected.append(
            {
                "attack": attack,
                "metric": row["metric"],
                "submetric": submetric,
                "expected": expected,
                "baseline": row["baseline"],
                "attacked": row["attacked"],
                "sensitive": row["sensitive"],
                "observed": "pass" if row["sensitive"] else "fail",
            }
        )
    output = {
        "status": "complete" if all(row["sensitive"] for row in selected) else "failed",
        "attacks": selected,
        "disclaimer": (
            "These attacks validate evaluator sensitivity; they are not used to "
            "increase the primary method score."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    table = [
        "# Compact FEVP Attack Sensitivity",
        "",
        "| Attack | Expected behavior | Baseline | Attacked | Observed |",
        "|---|---|---:|---:|---|",
    ]
    labels = {
        "move_untouched_room_1px": "Move an unedited room polygon",
        "inject_one_pixel_hole": "Inject a 1-pixel hole",
        "delete_adjacency_edge_f1": "Delete one adjacency edge",
        "relabel_one_node_type_f1": "Change one room label",
        "overlap_two_rooms": "Overlap two rooms",
    }
    for row in selected:
        table.append(
            f"| {labels[row['attack']]} | {row['expected']} | "
            f"{row['baseline']:.6f} | {row['attacked']:.6f} | {row['observed']} |"
        )
    table.extend(["", output["disclaimer"], ""])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": output["status"], "attacks": len(selected)}, indent=2))
    raise SystemExit(0 if output["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

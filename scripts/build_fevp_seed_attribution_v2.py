#!/usr/bin/env python3
"""Show five-seed consistency of edit-gating leakage attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outside",
        type=Path,
        default=ROOT / "outputs/reports/fevp_outside_layers_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_seed_attribution_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_seed_attribution_v2.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.outside.read_text(encoding="utf-8"))
    rows = []
    for seed, payload in sorted(
        source["summary"]["by_seed"].items(),
        key=lambda item: int(item[0]),
    ):
        strict = payload["strict"]
        no_gate = payload["no_edit_gating"]
        rows.append(
            {
                "seed": int(seed),
                "strict_room_exact": strict["outside_room_level_exact"],
                "strict_violation": strict["locality_violation_rate"],
                "no_gate_room_exact": no_gate["outside_room_level_exact"],
                "no_gate_violation": no_gate["locality_violation_rate"],
                "gate_room_gain": (
                    strict["outside_room_level_exact"]
                    - no_gate["outside_room_level_exact"]
                ),
                "gate_violation_reduction": (
                    no_gate["locality_violation_rate"]
                    - strict["locality_violation_rate"]
                ),
            }
        )
    report = {
        "status": "complete",
        "seeds": rows,
        "ranges": {
            "no_gate_room_exact": [
                min(row["no_gate_room_exact"] for row in rows),
                max(row["no_gate_room_exact"] for row in rows),
            ],
            "no_gate_violation": [
                min(row["no_gate_violation"] for row in rows),
                max(row["no_gate_violation"] for row in rows),
            ],
            "gate_room_gain": [
                min(row["gate_room_gain"] for row in rows),
                max(row["gate_room_gain"] for row in rows),
            ],
        },
        "checks": {
            "strict_exact_all_seeds": all(
                row["strict_room_exact"] == 1.0 for row in rows
            ),
            "strict_violation_zero_all_seeds": all(
                row["strict_violation"] == 0.0 for row in rows
            ),
            "no_gate_leaks_all_seeds": all(
                row["no_gate_room_exact"] < 0.55
                and row["no_gate_violation"] > 0.0
                for row in rows
            ),
            "gate_gain_positive_all_seeds": all(
                row["gate_room_gain"] > 0.0
                and row["gate_violation_reduction"] > 0.0
                for row in rows
            ),
        },
        "claim": (
            "No-gate leakage is consistent in direction across all five seeds "
            "and is attributed to removal of edit gating under otherwise matched "
            "rows, outputs, and evaluator."
        ),
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# Five-Seed Gating Attribution",
        "",
        "| Seed | Strict room exact | Strict violation | No-gate room exact | No-gate violation |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['seed']} | {row['strict_room_exact']:.6f} | "
            f"{row['strict_violation']:.9f} | "
            f"{row['no_gate_room_exact']:.6f} | "
            f"{row['no_gate_violation']:.9f} |"
        )
    table.extend(["", report["claim"], ""])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

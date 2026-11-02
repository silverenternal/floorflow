#!/usr/bin/env python3
"""Compute the per-case oracle bound of existing wall experts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    base_path = ROOT / "outputs/reports/predicted_control_wallgap002_val128_v1.json"
    host_path = (
        ROOT
        / "outputs/reports/cad_wall_candidate_hostonly_screen512_seed19_val128_v1.json"
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))["cases"]
    host_payload = json.loads(host_path.read_text(encoding="utf-8"))
    host = {row["case"]: row for row in host_payload["per_case"]["control"]}
    rows = []
    for row in base:
        alternative = host.get(row["case"])
        if alternative is None:
            continue
        rows.append(
            {
                "case": row["case"],
                "wall_gap002_f1": row["fpeval_wall_f1"],
                "host_only_f1": alternative["wall"],
                "oracle_f1": max(row["fpeval_wall_f1"], alternative["wall"]),
            }
        )
    if not rows:
        raise ValueError("wall experts have no matched cases")
    report = {
        "schema_version": "wall-parallel-upper-bound-v1",
        "scope": "predicted-control val128 matched wall experts",
        "test_opened": False,
        "target_used_at_inference": False,
        "cases": len(rows),
        "wall_gap002_f1": sum(row["wall_gap002_f1"] for row in rows) / len(rows),
        "host_only_f1": sum(row["host_only_f1"] for row in rows) / len(rows),
        "oracle_f1": sum(row["oracle_f1"] for row in rows) / len(rows),
        "gain_over_wall_gap002": sum(
            row["oracle_f1"] - row["wall_gap002_f1"] for row in rows
        )
        / len(rows),
        "wall_gap002_wins": sum(
            row["wall_gap002_f1"] >= row["host_only_f1"] for row in rows
        ),
        "host_only_wins": sum(
            row["host_only_f1"] > row["wall_gap002_f1"] for row in rows
        ),
        "decision": "new_candidates_required" if (
            sum(row["oracle_f1"] for row in rows) / len(rows)
        ) < 0.58 else "router_worth_screening",
        "sources": {
            "wall_gap002": str(base_path.relative_to(ROOT)),
            "host_only": str(host_path.relative_to(ROOT)),
        },
        "rows": rows,
    }
    output = ROOT / "outputs/reports/wall_parallel_upper_bound_val128_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "cases",
        "wall_gap002_f1",
        "host_only_f1",
        "oracle_f1",
        "gain_over_wall_gap002",
        "decision",
    )}, indent=2))


if __name__ == "__main__":
    main()

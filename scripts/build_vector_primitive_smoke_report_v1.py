#!/usr/bin/env python3
"""Summarize direct-query and proposal-residual vector smoke experiments."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECT = ROOT / "outputs/train/vector_primitive_smoke_v3_anchored/report.json"
REFINER = ROOT / "outputs/train/vector_proposal_refiner_smoke_v1/report.json"
OUTPUT = ROOT / "outputs/reports/vector_primitive_smoke_v1.json"
TABLE = ROOT / "paper/tables/vector_primitive_smoke_v1.md"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    direct = _read(DIRECT)
    refiner = _read(REFINER)
    direct_final = direct["history"][-1]["val"]
    refiner_final = refiner["history"][-1]["val"]
    reductions = {
        family: (
            refiner_final[f"{family}_baseline_mae"]
            - refiner_final[f"{family}_refined_mae"]
        )
        / max(1.0e-12, refiner_final[f"{family}_baseline_mae"])
        for family in ("room", "wall", "opening")
    }
    checks = {
        **refiner["checks"],
        "direct_query_rejected_as_main_route": (
            direct_final["room_f1"] < 0.2
            and direct_final["wall_f1"] < 0.2
            and direct_final["opening_f1"] < 0.2
        ),
        "all_refiner_mae_reduced": all(
            value > 0.0 for value in reductions.values()
        ),
    }
    report = {
        "status": "complete" if all(checks.values()) else "needs_iteration",
        "direct_query": {
            "checkpoint_report": str(DIRECT.relative_to(ROOT)),
            "val": direct_final,
            "decision": (
                "Kept as research branch; existence F1 is insufficient."
            ),
        },
        "proposal_residual": {
            "checkpoint_report": str(REFINER.relative_to(ROOT)),
            "val": refiner_final,
            "relative_mae_reduction": reductions,
            "decision": (
                "Retained as main vector route because primitive identity and "
                "coverage come from the existing readout."
            ),
        },
        "checks": checks,
        "claim_boundary": (
            "Architecture smoke only. These 256/128 results do not support a "
            "test-set SOTA claim."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Vector Primitive Smoke v1",
        "",
        "Architecture validation only; not a test-set SOTA result.",
        "",
        "| Route | Room | Wall | Opening |",
        "|---|---:|---:|---:|",
        (
            "| Direct query F1 | "
            f"{direct_final['room_f1']:.6f} | "
            f"{direct_final['wall_f1']:.6f} | "
            f"{direct_final['opening_f1']:.6f} |"
        ),
        (
            "| Proposal residual relative MAE reduction | "
            f"{reductions['room']:.4%} | "
            f"{reductions['wall']:.4%} | "
            f"{reductions['opening']:.4%} |"
        ),
        "",
        report["claim_boundary"],
    ]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()

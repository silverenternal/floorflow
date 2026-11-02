"""Summarize matched residential constraint-flow screens without cherry-picking."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "combined": ROOT / "outputs/reports/residential_constraint_flow_screen_v1.json",
    "fevp": ROOT / "outputs/reports/residential_constraint_flow_fevp_screen_v2.json",
    "ged": ROOT / "outputs/reports/residential_constraint_flow_ged_screen_v1.json",
}
OUTPUT = ROOT / "outputs/reports/residential_constraint_flow_summary_v1.json"


def main() -> None:
    screens = {}
    hard_exact = True
    promoted = []
    for name, path in SOURCES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        comparisons = payload["comparisons"]
        quality = payload["quality_checks"]
        best_trend = max(
            comparisons.values(),
            key=lambda row: float(row["mean_delta"]),
        )
        hard_exact &= bool(payload["checks"]["hard_valid_regions_exact"])
        if any(quality.values()):
            promoted.append(name)
        screens[name] = {
            "status": payload["status"],
            "quality_checks": quality,
            "best_mean_trend": {
                "mean_delta": best_trend["mean_delta"],
                "ci95": [best_trend["ci_low"], best_trend["ci_high"]],
            },
        }
    report = {
        "schema_version": "residential-constraint-flow-summary-v1",
        "scope": "residential_floorplan_only",
        "test_opened": False,
        "screens": screens,
        "checks": {
            "all_screens_hard_valid_exact": hard_exact,
            "at_least_one_proxy_promoted": bool(promoted),
        },
        "promoted_terms": promoted,
        "decision": (
            "promote"
            if promoted
            else "reject_soft_fevp_and_soft_ged_as_main_training_terms"
        ),
        "claim_boundary": (
            "The negative screen is retained as evidence; soft proxies are "
            "not reported as hard FEVP or hard GED."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT.relative_to(ROOT)), "decision": report["decision"]}))
    if not hard_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

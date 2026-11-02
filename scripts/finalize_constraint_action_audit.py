"""Recompute audit checks and record the downstream curation policy."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/reports/residential_constraint_action_audit_v1.json"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    splits = report["splits"]
    checks = {
        "train_val_only": set(splits) == {"train", "val"},
        "nine_candidates_per_group": all(
            result["candidates"] == 9 * result["groups"]
            for result in splits.values()
        ),
        "positive_hard_pass_rate_at_least_0_98": all(
            result["hard_flags_all_pass"].get("positive", 0)
            / max(result["roles"].get("positive", 0), 1)
            >= 0.98
            for result in splits.values()
        ),
        "legal_alternative_hard_pass_rate_at_least_0_98": all(
            result["hard_flags_all_pass"].get("legal_alternative", 0)
            / max(result["roles"].get("legal_alternative", 0), 1)
            >= 0.98
            for result in splits.values()
        ),
        "every_family_present": all(
            result["families"].get(family, 0) > 0
            for result in splits.values()
            for family in report["splits"]["train"]["families"]
        ),
        "hard_mask_recorded_for_every_role": all(
            set(result["hard_mask_pass_rate"])
            == {"positive", "legal_alternative", "near_violation"}
            for result in splits.values()
        ),
    }
    report["checks"] = checks
    report["status"] = "passed" if all(checks.values()) else "failed"
    report["downstream_policy"] = (
        "drop every group with an infeasible oracle positive or legal "
        "alternative; model-training positives are hard-feasible by construction"
    )
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

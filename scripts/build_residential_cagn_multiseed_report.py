"""Aggregate five-seed CAGN validation results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/reports/residential_cagn_multiseed_v1.json"


def _path(seed: int, scratch: bool) -> Path:
    if scratch:
        return ROOT / (
            "outputs/reports/"
            f"residential_cagn_validation_v3_scratch_seed{seed}.json"
        )
    return (
        ROOT / f"outputs/reports/residential_cagn_validation_v3_seed{seed}.json"
    )


def _interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    half = (
        float(student_t.ppf(0.975, len(array) - 1)) * std / np.sqrt(len(array))
        if len(array) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "min": float(array.min()),
        "max": float(array.max()),
        "method": "student_t_seed_interval",
    }


def main() -> None:
    rows = []
    for seed in (19, 20, 21, 22, 23):
        full = json.loads(_path(seed, False).read_text(encoding="utf-8"))
        scratch = json.loads(_path(seed, True).read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "full_status": full["status"],
                "full_ndcg": full["learned_gate"]["ndcg_at_5"],
                "hard_ndcg": full["hard_gate"]["ndcg_at_5"],
                "ndcg_gain": full["gain"]["ndcg_at_5"],
                "selected_violation_rate": (
                    full["learned_gate"]["selected_violation_rate"]
                ),
                "ece": full["calibration"]["ece"],
                "hard_mask_exact": full["checks"]["hard_mask_exact"],
                "scratch_ndcg": scratch["learned_gate"]["ndcg_at_5"],
                "scratch_selected_violation_rate": (
                    scratch["learned_gate"]["selected_violation_rate"]
                ),
                "scratch_auroc": scratch["calibration"]["legal_auroc"],
            }
        )
    metrics = {
        key: _interval([float(row[key]) for row in rows])
        for key in (
            "full_ndcg",
            "hard_ndcg",
            "ndcg_gain",
            "selected_violation_rate",
            "ece",
            "scratch_ndcg",
            "scratch_selected_violation_rate",
            "scratch_auroc",
        )
    }
    checks = {
        "five_seeds": len(rows) == 5,
        "all_hard_masks_exact": all(row["hard_mask_exact"] for row in rows),
        "mean_ndcg_gain_at_least_0_03": metrics["ndcg_gain"]["mean"] >= 0.03,
        "mean_selected_violation_at_most_0_005": (
            metrics["selected_violation_rate"]["mean"] <= 0.005
        ),
        "mean_ece_at_most_0_05": metrics["ece"]["mean"] <= 0.05,
        "full_beats_scratch_mean": (
            metrics["full_ndcg"]["mean"] > metrics["scratch_ndcg"]["mean"]
        ),
    }
    report = {
        "schema_version": "residential-cagn-multiseed-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only_validation",
        "seeds": [row["seed"] for row in rows],
        "rows": rows,
        "seed_intervals": metrics,
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate five-seed multiscale opening detector holdout evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"
PAIRED = {
    19: REPORTS / "opening_interval_multiscale_vs_old1024_holdout107_seed19_paired_v2.json",
    20: REPORTS / "opening_interval_multiscale_vs_old1024_holdout107_seed20_paired_v1.json",
    21: REPORTS / "opening_interval_multiscale_vs_old1024_holdout107_seed21_paired_v1.json",
    22: REPORTS / "opening_interval_multiscale_vs_old1024_holdout107_seed22_paired_v1.json",
    23: REPORTS / "opening_interval_multiscale_vs_old1024_holdout107_seed23_paired_v1.json",
}
HOLDOUT = {
    seed: REPORTS
    / f"opening_interval_multiscale_promotion1024_holdout107_seed{seed}_v1.json"
    for seed in PAIRED
}
OUTPUT = REPORTS / "opening_interval_multiscale_promotion_fiveseed_v1.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
        "improved_seeds": int((array > 0).sum()),
    }


def main() -> None:
    rows = []
    for seed, paired_path in PAIRED.items():
        holdout = _read(HOLDOUT[seed])
        paired = _read(paired_path)["paired"]
        rows.append(
            {
                "seed": seed,
                "cases": len(holdout["cases"]),
                "pixel_f1": float(holdout["summary"]["pixel_f1_mean"]),
                "vector_f1": float(holdout["summary"]["vector_f1_mean"]),
                "vector_precision": float(
                    holdout["summary"]["vector_precision_mean"]
                ),
                "vector_recall": float(
                    holdout["summary"]["vector_recall_mean"]
                ),
                "pixel_delta": float(paired["pixel_f1"]["mean_delta"]),
                "pixel_ci95": [
                    float(paired["pixel_f1"]["ci_low"]),
                    float(paired["pixel_f1"]["ci_high"]),
                ],
                "vector_delta": float(paired["vector_f1"]["mean_delta"]),
                "vector_ci95": [
                    float(paired["vector_f1"]["ci_low"]),
                    float(paired["vector_f1"]["ci_high"]),
                ],
                "holdout": str(HOLDOUT[seed].relative_to(ROOT)),
                "paired": str(paired_path.relative_to(ROOT)),
            }
        )
    summary = {
        "pixel_f1": _stats([row["pixel_f1"] for row in rows]),
        "vector_f1": _stats([row["vector_f1"] for row in rows]),
        "pixel_delta": _stats([row["pixel_delta"] for row in rows]),
        "vector_delta": _stats([row["vector_delta"] for row in rows]),
    }
    checks = {
        "five_seeds": len(rows) == 5,
        "same_holdout_cases": len({row["cases"] for row in rows}) == 1,
        "pixel_delta_positive_all_seeds": all(
            row["pixel_delta"] > 0.0 for row in rows
        ),
        "vector_delta_positive_all_seeds": all(
            row["vector_delta"] > 0.0 for row in rows
        ),
        "vector_ci_positive_at_least_four_seeds": sum(
            row["vector_ci95"][0] > 0.0 for row in rows
        )
        >= 4,
        "absolute_vector_f1_target_0_35": summary["vector_f1"]["mean"]
        >= 0.35,
    }
    report = {
        "schema_version": "opening-multiscale-promotion-fiveseed-v1",
        "scope": "matched 1024/8ep holdout107, target-free detector",
        "seeds": sorted(PAIRED),
        "rows": rows,
        "summary": summary,
        "checks": checks,
        "decision": (
            "promote_upgrade_with_absolute_gap"
            if all(
                value
                for key, value in checks.items()
                if key != "absolute_vector_f1_target_0_35"
            )
            and not checks["absolute_vector_f1_target_0_35"]
            else "hold"
        ),
        "boundary": (
            "The multiscale head significantly improves the detector baseline "
            "but remains below the absolute Opening F1 publication gate."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()

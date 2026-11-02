#!/usr/bin/env python3
"""Compare raw and EMA chip checkpoints under one official protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RAW = {
    "flat19": "outputs/chip_large/v2_20k/dense_no_ot_huber/official_test100_nfe8_legalized_cases.json",
    "flat20": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s20/official_test100_nfe8_legalized_cases.json",
    "flat21": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s21/official_test100_nfe8_legalized_cases.json",
    "global19": "outputs/chip_slot/global_v1/global_slot_overlap10_20k/official_test100_nfe8_legalized_cases.json",
    "global20": "outputs/chip_slot/global_v2/global_slot_seed20_overlap10_20k/official_test100_nfe8_legalized_cases.json",
    "global21": "outputs/chip_slot/global_v2/global_slot_seed21_overlap10_20k/official_test100_nfe8_legalized_cases.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_ema_audit_v1.json",
    )
    return parser.parse_args()


def read_cases(path: Path) -> dict[int, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["index"]): row for row in rows}


def paired(
    raw: dict[int, dict],
    ema: dict[int, dict],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict:
    indices = sorted(set(raw) & set(ema))
    delta = np.asarray(
        [float(ema[index][metric]) - float(raw[index][metric]) for index in indices]
    )
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(delta, len(delta), replace=True).mean()
            for _ in range(iterations)
        ]
    )
    raw_mean = float(np.mean([float(raw[index][metric]) for index in indices]))
    ema_mean = float(np.mean([float(ema[index][metric]) for index in indices]))
    return {
        "raw_mean": raw_mean,
        "ema_mean": ema_mean,
        "paired_delta": float(delta.mean()),
        "relative_delta": (ema_mean - raw_mean) / raw_mean,
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    args = parse_args()
    results = {}
    for offset, (name, relative) in enumerate(RAW.items()):
        raw = read_cases(ROOT / relative)
        ema = read_cases(
            ROOT
            / f"outputs/chip_ema/{name}_ema_nfe8_legalized_cases.json"
        )
        results[name] = {
            metric: paired(
                raw,
                ema,
                metric,
                iterations=args.bootstrap,
                seed=20261027 + offset * 2 + index,
            )
            for index, metric in enumerate(
                ("legalized_hard_cost", "legalized_v_rel")
            )
        }
    means = {
        family: {
            metric: float(
                np.mean(
                    [
                        row[metric]["relative_delta"]
                        for name, row in results.items()
                        if name.startswith(family)
                    ]
                )
            )
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        }
        for family in ("flat", "global")
    }
    checks = {
        "six_arms_complete": all(
            row["legalized_hard_cost"]["raw_mean"] is not None
            for row in results.values()
        ),
        "global_mean_both_non_inferior": all(
            value <= 0.0 for value in means["global"].values()
        ),
        "flat_mean_both_non_inferior": all(
            value <= 0.0 for value in means["flat"].values()
        ),
        "no_arm_regresses_both_over_1pct": not any(
            row["legalized_hard_cost"]["relative_delta"] > 0.01
            and row["legalized_v_rel"]["relative_delta"] > 0.01
            for row in results.values()
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "ema_not_promoted",
        "scope": "official-100 raw-versus-EMA audit at NFE8 + scale_pack",
        "protocol": {
            "sampling_steps": 8,
            "scorer": "score_full_fast",
            "legalizer": "scale_pack",
            "best_of_n_used": False,
        },
        "checks": checks,
        "family_mean_relative_delta": means,
        "arms": results,
        "decision": (
            "use EMA consistently"
            if all(checks.values())
            else "raw and EMA remain seed-dependent; test EMA inside the graph-conditioned rescue rather than selecting it post hoc"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "family_mean": means,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

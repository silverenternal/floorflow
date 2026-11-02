#!/usr/bin/env python3
"""Verify the bounded-mechanism invariant and measured envelopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.model import decode_boxes_batch

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_bound_invariant_cpu_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def case_paths() -> dict[str, dict[int, Path]]:
    root = ROOT / "outputs/chip_hierarchy/v10_three_seed/reproducible_v1"
    return {
        "candidate": {
            seed: root / f"candidate_seed{seed}_ema_primary_cases.json"
            for seed in (19, 20, 21)
        },
        "control": {
            seed: root / f"baseline_seed{seed}_ema_primary_cases.json"
            for seed in (19, 20, 21)
        },
    }


def load_rows(paths: dict[int, Path]) -> list[dict]:
    return [
        row
        for path in paths.values()
        for row in read(path)["nfe4_refine400"]
    ]


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    points = np.quantile(array, (0.5, 0.9, 0.95))
    return {
        "cases": int(array.size),
        "p50": float(points[0]),
        "p90": float(points[1]),
        "p95": float(points[2]),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def envelope(rows: list[dict]) -> dict[str, dict]:
    fields = (
        "raw_overlap_pairs",
        "raw_boundary",
        "legalized_overlap_pairs",
        "legalized_area_violations",
        "legalized_fixed_violations",
        "legalized_preplaced_violations",
        "legalized_aspect_violations",
    )
    return {
        field: distribution([float(row[field]) for row in rows])
        for field in fields
    }


def exact_area_audit() -> dict:
    generator = torch.Generator().manual_seed(20261028)
    areas = torch.exp(2.0 * torch.randn(8, 97, generator=generator))
    latent = torch.randn(8, 97, 3, generator=generator) * 3.0
    boxes = decode_boxes_batch(latent, areas)
    decoded_area = boxes[..., 2] * boxes[..., 3]
    error = (decoded_area - areas).abs()
    relative = error / areas.clamp_min(1.0e-12)
    return {
        "cases": 8,
        "tokens": int(areas.numel()),
        "max_absolute_error": float(error.max()),
        "max_relative_error": float(relative.max()),
        "pass": float(relative.max()) < 2.0e-6,
    }


def main() -> None:
    args = parse_args()
    rows = {
        arm: load_rows(paths)
        for arm, paths in case_paths().items()
    }
    envelopes = {arm: envelope(values) for arm, values in rows.items()}
    area = exact_area_audit()
    universal_legalized = all(
        bool(row["legalized_feasible"])
        and int(row["legalized_overlap_pairs"]) == 0
        and int(row["legalized_fixed_violations"]) == 0
        and int(row["legalized_preplaced_violations"]) == 0
        and int(row["legalized_area_violations"]) == 0
        and int(row["legalized_aspect_violations"]) == 0
        for arm_rows in rows.values()
        for row in arm_rows
    )
    raw_envelope_improves = {}
    for field in ("raw_overlap_pairs", "raw_boundary"):
        raw_envelope_improves[field] = {
            quantile: (
                envelopes["candidate"][field][quantile]
                < envelopes["control"][field][quantile]
            )
            for quantile in ("p50", "p90", "p95", "max")
        }
    checks = {
        "exact_area_decoder": area["pass"],
        "all_scored_outputs_checker_feasible": universal_legalized,
        "one_to_one_denominators": (
            envelopes["candidate"]["raw_overlap_pairs"]["cases"]
            == envelopes["control"]["raw_overlap_pairs"]["cases"]
            == 300
        ),
        "candidate_has_partial_raw_overlap_envelope_gain": (
            raw_envelope_improves["raw_overlap_pairs"]["p50"]
            or raw_envelope_improves["raw_overlap_pairs"]["p90"]
            or raw_envelope_improves["raw_overlap_pairs"]["p95"]
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "bounded-mechanism invariant and available empirical envelopes",
        "contract": "docs/chip_bounded_mechanism_contract_v1.md",
        "theorem_scope": {
            "proved": [
                "exact-area decoder preserves target area up to finite tolerance",
                "checker-accepted legalized output has zero hard violations",
            ],
            "not_proved": [
                "lower FloorSet cost",
                "lower routed wirelength or PPA",
                "uniform per-case or per-seed improvement",
            ],
            "mechanism_label": "empirical_bounded_mechanism",
        },
        "exact_area": area,
        "envelopes": envelopes,
        "envelope_direction": raw_envelope_improves,
        "pending_missing_envelopes": [
            "legalization displacement p50/p90/p95/max",
            "legalizer iteration count and fallback",
            "pin-to-boundary conflict rate",
            "routed congestion and timing envelopes",
        ],
        "checks": checks,
        "decision": {
            "invariant_verified": all(checks.values()),
            "mechanism_claim": "empirical_bounded_mechanism",
            "next": "MECH-TRACE must fill displacement and conflict envelopes.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stratify existing post-fix portfolio evidence by complexity and density."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.training_runtime import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_complexity_tail_audit_v1.json",
    )
    return parser.parse_args()


def _reports() -> dict[int, tuple[Path, Path]]:
    return {
        19: (
            ROOT / "outputs/reports/chip_m48_portfolio_official100_postfix_seed19.json",
            ROOT / "outputs/reports/chip_trinity_d192_nfe1_ref1600_official100_v1.json",
        ),
        20: (
            ROOT / "outputs/reports/chip_m48_portfolio_official100_seed20.json",
            ROOT / "outputs/reports/chip_trinity_d192_nfe1_ref1600_official100_seed20.json",
        ),
        21: (
            ROOT / "outputs/reports/chip_m48_portfolio_official100_seed21.json",
            ROOT / "outputs/reports/chip_trinity_d192_nfe1_ref1600_official100_seed21.json",
        ),
    }


def _bucket(value: float, cuts: tuple[float, ...]) -> str:
    labels = [f"<={cuts[0]:.2f}"]
    labels.extend(
        f"({cuts[index]:.2f},{cuts[index + 1]:.2f}]"
        for index in range(len(cuts) - 1)
    )
    labels.append(f">{cuts[-1]:.2f}")
    for index, cut in enumerate(cuts):
        if value <= cut:
            return labels[index]
    return labels[-1]


def _aggregate(rows: list[dict], family: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[family]].append(row)
    result = {}
    for name, members in sorted(grouped.items()):
        candidate_cost = float(np.mean([row["cost"] for row in members]))
        candidate_v_rel = float(np.mean([row["v_rel"] for row in members]))
        candidate_b2b = float(np.mean([row["b2b_hpwl_gap"] for row in members]))
        anchor_cost = float(np.mean([row["anchor_cost"] for row in members]))
        anchor_v_rel = float(np.mean([row["anchor_v_rel"] for row in members]))
        anchor_b2b = float(np.mean([row["anchor_b2b"] for row in members]))
        result[name] = {
            "cases": len(members),
            "candidate_cost": candidate_cost,
            "candidate_v_rel": candidate_v_rel,
            "anchor_cost": anchor_cost,
            "anchor_v_rel": anchor_v_rel,
            "cost_relative": candidate_cost / anchor_cost - 1.0,
            "v_rel_relative": (
                candidate_v_rel / anchor_v_rel - 1.0
                if anchor_v_rel > 0.0
                else 0.0
            ),
            "b2b_relative": (
                candidate_b2b / anchor_b2b - 1.0
                if anchor_b2b > 0.0
                else 0.0
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    store = PackedCaseStore(args.cache)
    rows = []
    for seed, (portfolio_path, anchor_path) in _reports().items():
        portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        anchor_rows = {int(row["index"]): row for row in anchor["rows"]}
        for row in portfolio["rows"]:
            index = int(row["index"])
            if index not in anchor_rows:
                continue
            batch = store.batch([index])
            count = int(batch["n"])
            b2b_density = int(batch["n_b2b"][0]) / count
            pin_density = int(batch["n_p2b"][0]) / count
            boundary = bool(batch["constraints"][0, :, 4].ne(0).any())
            selected = row["selected"]["stages"]["legalized"]
            reference = anchor_rows[index]
            rows.append(
                {
                    "seed": seed,
                    "index": index,
                    "blocks_bucket": (
                        "<=40" if count <= 40
                        else "41-60" if count <= 60
                        else "61-80" if count <= 80
                        else "81-100" if count <= 100
                        else "101-120"
                    ),
                    "b2b_bucket": _bucket(b2b_density, (4.0, 10.0)),
                    "pin_bucket": _bucket(pin_density, (3.0, 9.0)),
                    "boundary_bucket": "boundary" if boundary else "interior",
                    "cost": float(selected["cost"]),
                    "v_rel": float(selected["v_rel"]),
                    "b2b_hpwl_gap": float(selected["b2b_hpwl_gap"]),
                    "anchor_cost": float(
                        reference.get(
                            "legalized_hard_cost",
                            reference.get("cost"),
                        )
                    ),
                    "anchor_v_rel": float(
                        reference.get(
                            "legalized_v_rel",
                            reference.get("v_rel"),
                        )
                    ),
                    "anchor_b2b": float(
                        reference.get("legalized_b2b_hpwl_gap", 0.0)
                    ),
                }
            )
    report = {
        "version": "chip-complexity-tail-audit-v1",
        "status": "cpu_diagnostic_complete",
        "scope": "Existing post-fix portfolio rows; candidate uses 2x generation budget.",
        "cases": len(rows),
        "strata": {
            family: _aggregate(rows, family)
            for family in (
                "blocks_bucket",
                "b2b_bucket",
                "pin_bucket",
                "boundary_bucket",
            )
        },
        "boundary": "Descriptive evidence only; no new inference or matched-budget claim.",
    }
    write_json_atomic(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()

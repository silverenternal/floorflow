#!/usr/bin/env python3
"""Formalize the PlanC chip and house transfer into a cross-domain table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chip-report",
        type=Path,
        default=ROOT / "outputs/reports/plan_c_global_slot_seeds_v1.json",
    )
    parser.add_argument(
        "--house-report",
        type=Path,
        default=ROOT / "outputs/reports/plan_c_house_global_slot_seeds_v1.json",
    )
    parser.add_argument(
        "--claim-freeze",
        type=Path,
        default=ROOT / "outputs/reports/claim_freeze_reframe_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/cross_domain_ablation_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chip = read(args.chip_report)
    house = read(args.house_report)
    freeze = read(args.claim_freeze)
    sparse_arms = {}
    for arm in ("sparse16_no_ot", "sparse16_ot"):
        path = ROOT / f"outputs/chip_large/v2_20k/{arm}/official_test100_nfe8_legalized.json"
        if path.is_file():
            payload = read(path)
            sparse_arms[arm] = {
                "legalized_hard_cost": payload.get("legalized_hard_cost"),
                "legalized_v_rel": payload.get("legalized_v_rel"),
                "legalized_feasible_fraction": payload.get(
                    "legalized_feasible_fraction"
                ),
            }
    shared = [
        "permutation-equivariant set flow",
        "global-latent transport with correspondence decoupling",
        "boundary/outline conditioning tokens",
        "exact-area box decode for token sets",
        "paired-bootstrap seed gate",
    ]
    domain_specific = {
        "house": "raster room masks and per-room box targets",
        "chip": "FloorSet block constraints, nets, pins, and scale_pack legalizer",
    }
    report = {
        "status": "pass"
        if chip.get("gate", {}).get("status") == "pass"
        and house.get("gate", {}).get("status") == "pass"
        else "fail",
        "hypothesis": (
            "Correspondence-decoupled conditional flow transfers across "
            "layout domains when the shared primitive is a constrained token-set flow."
        ),
        "shared_primitives": shared,
        "domain_specific": domain_specific,
        "chip": {
            "task": "FloorSet block floorplanning with Trinity scorer and legalizer",
            "metric": "legalized hard cost and V_rel",
            "summary": chip.get("official_seed_summary"),
            "length_bands": [
                row.get("official_by_length")
                for row in chip.get("seeds", [])
            ],
            "gate": chip.get("gate"),
            "sparse_attention_official": sparse_arms,
        },
        "house": {
            "task": "CubiCasa room bounding-box generation",
            "metric": "paired bbox IoU and outside fraction",
            "summary": house.get("summary"),
            "gate": house.get("gate"),
        },
        "architecture_caveat": (
            "The chip arm is the global-latent/slot variant, not the full "
            "raster hierarchical pipeline; shared primitives rather than "
            "identical heads are the transfer claim."
        ),
        "claim_alignment": freeze["chip_transfer"]["gate"],
        "best_of_n_used": False,
        "target_derived_control": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

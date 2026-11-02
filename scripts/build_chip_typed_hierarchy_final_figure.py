#!/usr/bin/env python3
"""Build the final EMA NFE/refiner figure for the typed chip hierarchy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_chip_typed_hierarchy_diagnostics import plot_pareto

DATA = ROOT / "outputs/chip_hierarchy/v10_three_seed/reproducible_v1"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    candidate_nfe = {
        int(key.removeprefix("nfe").split("_")[0]): value
        for key, value in read(DATA / "candidate_seed19_ema_nfe.json")[
            "results"
        ].items()
    }
    baseline_nfe = {
        int(key.removeprefix("nfe").split("_")[0]): value
        for key, value in read(DATA / "baseline_seed19_ema_nfe.json")[
            "results"
        ].items()
    }
    candidate_refiner = {
        int(key.rsplit("refine", 1)[1]): value
        for key, value in read(
            DATA / "candidate_seed19_ema_refiner_short.json"
        )["results"].items()
    }
    candidate_refiner[400] = read(
        DATA / "candidate_seed19_ema_primary.json"
    )["results"]["nfe4_refine400"]
    baseline_primary = read(
        DATA / "baseline_seed19_ema_primary.json"
    )["results"]["nfe4_refine400"]
    paths = plot_pareto(
        candidate_nfe,
        baseline_nfe,
        candidate_refiner,
        baseline_primary,
        ROOT / "outputs/figures/chip_typed_hierarchy_final_pareto_v1",
    )
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()

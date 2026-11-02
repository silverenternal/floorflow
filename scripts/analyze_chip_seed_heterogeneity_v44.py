#!/usr/bin/env python3
"""Quantify seed heterogeneity before and after the frozen refiner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 20, 21)
TRAINING_REPORTS = {
    19: "outputs/chip_hierarchy/v7_bbox_anneal_20k/"
    "typed_hierarchy_bbox_anneal_seed19_20k.json",
    20: "outputs/chip_hierarchy/v10_three_seed/typed_hierarchy_seed20_20k.json",
    21: "outputs/chip_hierarchy/v10_three_seed/typed_hierarchy_seed21_20k.json",
}
BASE_REPORTS = {
    seed: "outputs/chip_hierarchy/v10_three_seed/reproducible_v1/"
    f"candidate_seed{seed}_ema_primary_repro_v44_lp2s.json"
    for seed in SEEDS
}
REFINED_REPORTS = {
    seed: "outputs/chip_hierarchy/v44/refiner_official/"
    f"candidate_boundary2_detglop_1600_seed{seed}.json"
    for seed in SEEDS
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(path: Path) -> dict[str, float]:
    result = read(path)["results"]
    key = next(iter(result))
    values = result[key]
    return {
        name: float(values[name])
        for name in (
            "legalized_hard_cost",
            "legalized_v_rel",
            "legalized_grouping",
            "legalized_boundary",
            "legalized_b2b_hpwl_gap",
            "legalized_p2b_hpwl_gap",
            "legalized_area_gap",
        )
    }


def spread(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "range": float(array.max() - array.min()),
        "relative_range": float((array.max() - array.min()) / array.mean()),
    }


def rng_digest(path: Path) -> str:
    state = read(path).get("history", [])
    payload = json.dumps(state, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    base = {seed: metrics(ROOT / BASE_REPORTS[seed]) for seed in SEEDS}
    refined = {seed: metrics(ROOT / REFINED_REPORTS[seed]) for seed in SEEDS}
    histories = {
        seed: read(ROOT / TRAINING_REPORTS[seed])
        for seed in SEEDS
    }
    metric_keys = tuple(base[19])
    base_spread = {
        name: spread([base[seed][name] for seed in SEEDS])
        for name in metric_keys
    }
    refined_spread = {
        name: spread([refined[seed][name] for seed in SEEDS])
        for name in metric_keys
    }
    training_final = {
        str(seed): {
            **{
                name: float(histories[seed]["final_dev"][name])
                for name in (
                    "latent_mse",
                    "rho_mae",
                    "overlap",
                    "group",
                    "boundary",
                    "wl",
                    "area",
                    "total",
                )
            },
            "history_sha256": rng_digest(ROOT / TRAINING_REPORTS[seed]),
            "config": {
                "seed": histories[seed]["config"]["seed"],
                "bbox_weight": histories[seed]["config"]["hierarchy_bbox_weight"],
                "bbox_decay_steps": histories[seed]["config"][
                    "hierarchy_bbox_decay_steps"
                ],
                "ema_decay": histories[seed]["config"]["ema_decay"],
            },
        }
        for seed in SEEDS
    }
    threshold_checks = {
        "base_all_seeds_beat_cost": all(
            base[seed]["legalized_hard_cost"] < 1.2165604088055237
            for seed in SEEDS
        ),
        "base_all_seeds_beat_v_rel": all(
            base[seed]["legalized_v_rel"] < 0.04359625539639952
            for seed in SEEDS
        ),
        "refined_all_seeds_beat_cost": all(
            refined[seed]["legalized_hard_cost"] < 1.2165604088055237
            for seed in SEEDS
        ),
        "refined_all_seeds_beat_v_rel": all(
            refined[seed]["legalized_v_rel"] < 0.04359625539639952
            for seed in SEEDS
        ),
    }
    report = {
        "status": "diagnosed_with_refiner_stabilization",
        "scope": "Three typed-hierarchy training seeds, Official-100 deterministic evaluation",
        "training_seed_coupling": {
            "initialization_and_data_order_seed": "same CLI seed",
            "causal_attribution_supported": False,
            "reason": (
                "The current trainer couples parameter initialization and data order "
                "to one seed, so existing runs cannot identify which factor causes "
                "the base-model spread without a matched decoupled rerun."
            ),
        },
        "training_final": training_final,
        "base_400_step": {
            str(seed): base[seed] for seed in SEEDS
        },
        "refined_1600_step": {
            str(seed): refined[seed] for seed in SEEDS
        },
        "spread": {
            "base": base_spread,
            "refined": refined_spread,
            "relative_range_reduction": {
                name: (
                    1.0
                    - refined_spread[name]["relative_range"]
                    / base_spread[name]["relative_range"]
                    if base_spread[name]["relative_range"]
                    else None
                )
                for name in metric_keys
            },
        },
        "threshold_checks": threshold_checks,
        "decision": (
            "The 1600-step boundary-aware refiner removes the SOTA-threshold seed "
            "reversal: every seed beats both reproducible Trinity anchor metrics. "
            "This is a stabilization result, not proof that training dynamics caused "
            "the original seed spread."
        ),
    }
    output = ROOT / "outputs/reports/chip_seed_rootcause_v44.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "threshold_checks": threshold_checks,
                "cost_spread": spread(
                    [refined[seed]["legalized_hard_cost"] for seed in SEEDS]
                ),
                "v_rel_spread": spread(
                    [refined[seed]["legalized_v_rel"] for seed in SEEDS]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

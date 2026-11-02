"""Component decomposition helpers for the matched chip comparator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (19, 20, 21)
# Mirrors third_party/Trinity/src/trinity/floorplan/scoring/cost.py.
ALPHA = 0.5
BETA = 2.0
COMPONENTS = {
    "b2b_connectivity": "legalized_b2b_hpwl_gap",
    "area_outline": "legalized_area_gap",
    "grouping": "legalized_grouping",
    "p2b_port": "legalized_p2b_hpwl_gap",
    "boundary": "legalized_boundary",
    "mib": "legalized_mib",
    "rho_aspect": "legalized_rho_mae",
}
GROUPS = {
    "b2b_connectivity": "structural",
    "area_outline": "geometric",
    "grouping": "structural",
    "p2b_port": "port",
    "boundary": "structural",
    "mib": "structural",
    "rho_aspect": "geometric",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(row: dict, name: str) -> float:
    key = f"legalized_{name}"
    return float(row[key] if key in row else row[name])


def optional_metric(row: dict, name: str) -> float | None:
    key = f"legalized_{name}"
    if key not in row and name not in row:
        return None
    return metric(row, name)


def soft_contributions(row: dict) -> dict[str, float]:
    total = float(
        metric(row, "grouping")
        + metric(row, "boundary")
        + metric(row, "mib")
    )
    denominator = row.get("legalized_soft_slots")
    if denominator is None:
        v_rel = metric(row, "v_rel")
        denominator = total / v_rel if total and v_rel else 1.0
    denominator = max(float(denominator), 1.0)
    return {
        "grouping_contribution": metric(row, "grouping") / denominator,
        "boundary_contribution": metric(row, "boundary") / denominator,
        "mib_contribution": metric(row, "mib") / denominator,
    }


def canonical_row(row: dict, *, arm: str) -> dict:
    if arm == "candidate":
        cost = row["legalized_hard_cost"]
        v_rel = row["legalized_v_rel"]
    else:
        cost = row.get("legalized_hard_cost", row["cost"])
        v_rel = row.get("legalized_v_rel", row["v_rel"])
    contributions = soft_contributions(row)
    hpwl = metric(row, "hpwl_gap")
    area = metric(row, "area_gap")
    reconstructed_cost = (
        1.0
        + ALPHA * (max(0.0, hpwl) + max(0.0, area))
    ) * float(np.exp(BETA * float(v_rel)))
    component_values = {
        key: value
        for key in COMPONENTS.values()
        if (
            value := optional_metric(
                row,
                key.removeprefix("legalized_"),
            )
        )
        is not None
    }
    return {
        "index": int(row["index"]),
        "cost": float(cost),
        "v_rel": float(v_rel),
        **contributions,
        **component_values,
        "soft_component_sum": sum(contributions.values()),
        "log_cost": float(np.log(float(cost))),
        "log_quality": float(
            np.log(
                1.0
                + ALPHA * (max(0.0, hpwl) + max(0.0, area))
            )
        ),
        "log_soft": float(BETA * float(v_rel)),
        "reconstructed_cost": reconstructed_cost,
    }


def averaged_rows(rows: list[dict], *, arm: str) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["index"]), []).append(
            canonical_row(row, arm=arm)
        )
    return [
        {
            key: float(np.mean([float(case[key]) for case in cases]))
            for key in cases[0]
        }
        | {"index": index}
        for index, cases in sorted(grouped.items())
    ]


def candidate_rows(payload: dict) -> list[dict]:
    rows = [
        row
        for row in payload["rows"]
        if row["arm"] == "candidate" and row["stage"] == "refine400"
    ]
    output = averaged_rows(rows, arm="candidate")
    if len(output) != 100:
        raise ValueError("expected 100 candidate official cases")
    return output


def trinity_seed_rows() -> dict[int, list[dict]]:
    return {
        seed: averaged_rows(
            read(
                ROOT
                / f"outputs/reports/trinity_d192_component_seed{seed}_v1.json"
            )["rows"],
            arm="trinity",
        )
        for seed in SEEDS
    }


def seed_average(by_seed: dict[int, list[dict]]) -> list[dict]:
    grouped = {}
    for rows in by_seed.values():
        for row in rows:
            grouped.setdefault(int(row["index"]), []).append(row)
    return [
        {
            key: float(np.mean([float(case[key]) for case in cases]))
            for key in cases[0]
        }
        | {"index": index}
        for index, cases in sorted(grouped.items())
    ]


def paired_relative(
    candidate: list[dict],
    baseline: list[dict],
    key: str,
    *,
    seed: int,
) -> dict:
    stats = paired_delta(
        candidate,
        baseline,
        key,
        iterations=4000,
        seed=seed,
        lower_is_better=True,
    )
    baseline_mean = float(
        np.mean([float(row[key]) for row in baseline])
    )
    candidate_mean = float(
        np.mean([float(row[key]) for row in candidate])
    )
    scale = abs(baseline_mean)
    stats["candidate_mean"] = candidate_mean
    stats["trinity_mean"] = baseline_mean
    stats["relative_gap"] = stats["mean_delta"] / scale if scale else None
    stats["relative_ci_low"] = stats["ci_low"] / scale if scale else None
    stats["relative_ci_high"] = stats["ci_high"] / scale if scale else None
    return stats


def component_report(
    candidate: list[dict],
    trinity: list[dict],
    *,
    seed: int,
) -> dict:
    return {
        name: paired_relative(
            candidate,
            trinity,
            key,
            seed=seed + index,
        )
        | {"group": GROUPS[name], "lower_is_better": True}
        for index, (name, key) in enumerate(COMPONENTS.items())
    }


def vrel_decomposition(
    candidate: list[dict],
    trinity: list[dict],
    *,
    seed: int,
) -> dict:
    aggregate_gap = float(
        np.mean([row["v_rel"] for row in candidate])
        - np.mean([row["v_rel"] for row in trinity])
    )
    output = {}
    for index, name in enumerate(("grouping", "boundary", "mib")):
        stats = paired_relative(
            candidate,
            trinity,
            f"{name}_contribution",
            seed=seed + index,
        )
        stats["share_of_aggregate_vrel_gap"] = (
            stats["mean_delta"] / aggregate_gap
            if aggregate_gap
            else None
        )
        output[name] = stats
    component_sum = float(
        sum(item["mean_delta"] for item in output.values())
    )
    return {
        "aggregate_vrel_gap": aggregate_gap,
        "contributions": output,
        "sum_component_gaps": component_sum,
        "reconstruction_error": aggregate_gap - component_sum,
    }


def cost_decomposition(
    candidate: list[dict],
    trinity: list[dict],
) -> dict:
    candidate_log = {
        key: float(np.mean([row[key] for row in candidate]))
        for key in ("log_cost", "log_quality", "log_soft")
    }
    trinity_log = {
        key: float(np.mean([row[key] for row in trinity]))
        for key in ("log_cost", "log_quality", "log_soft")
    }
    total = candidate_log["log_cost"] - trinity_log["log_cost"]
    quality = candidate_log["log_quality"] - trinity_log["log_quality"]
    soft = candidate_log["log_soft"] - trinity_log["log_soft"]
    return {
        "definition": (
            "log(cost)=log(1+ALPHA*(max(hpwl_gap,0)+max(area_gap,0)))"
            "+BETA*V_rel; overlap is zero in the retained rows"
        ),
        "alpha": ALPHA,
        "beta": BETA,
        "candidate_log_terms": candidate_log,
        "trinity_log_terms": trinity_log,
        "log_gap": total,
        "geometry_log_gap": quality,
        "soft_log_gap": soft,
        "reconstruction_error": total - quality - soft,
        "geometric_mean_cost_gap": float(np.exp(total) - 1.0),
        "geometry_share_of_log_gap": quality / total if total else None,
        "soft_share_of_log_gap": soft / total if total else None,
    }


def headline_audit(
    component_gap: dict,
    candidate: list[dict],
    trinity: list[dict],
) -> dict:
    primary_candidate = component_gap["main_comparison"]["candidate"]
    candidate_cost = float(np.mean([row["cost"] for row in candidate]))
    candidate_v_rel = float(np.mean([row["v_rel"] for row in candidate]))
    trinity_cost = float(np.mean([row["cost"] for row in trinity]))
    trinity_v_rel = float(np.mean([row["v_rel"] for row in trinity]))
    return {
        "predeclared_gap": {
            "cost": component_gap["main_comparison"][
                "legacy_reference"
            ]["candidate_relative_gap"]["cost_relative"],
            "v_rel": component_gap["main_comparison"][
                "legacy_reference"
            ]["candidate_relative_gap"]["v_rel_relative"],
        },
        "published_primary_candidate": primary_candidate,
        "component_row_candidate": {
            "cost": candidate_cost,
            "v_rel": candidate_v_rel,
        },
        "component_row_trinity": {
            "cost": trinity_cost,
            "v_rel": trinity_v_rel,
        },
        "component_row_gap": {
            "cost": candidate_cost / trinity_cost - 1.0,
            "v_rel": candidate_v_rel / trinity_v_rel - 1.0,
        },
        "candidate_source_residual": {
            "cost": candidate_cost / primary_candidate["cost"] - 1.0,
            "v_rel": candidate_v_rel / primary_candidate["v_rel"] - 1.0,
        },
    }


def survival_report(
    candidate: list[dict],
    trinity: list[dict],
    mechanism: dict,
) -> dict:
    output = {}
    soft_names = {"grouping", "boundary", "mib"}
    for name, raw_key in COMPONENTS.items():
        key = f"{name}_contribution" if name in soft_names else raw_key
        mechanism_key = (
            f"legalized_{name}_vrel"
            if name in soft_names
            else raw_key
        )
        candidate_mean = float(
            np.mean([row[key] for row in candidate])
        )
        trinity_mean = float(
            np.mean([row[key] for row in trinity])
        )
        global_mean = (
            float(
                mechanism["refined_pooled"][mechanism_key]["baseline_mean"]
            )
            if mechanism_key in mechanism["refined_pooled"]
            else None
        )
        beats_global = (
            candidate_mean < global_mean if global_mean is not None else None
        )
        reaches_trinity = candidate_mean <= trinity_mean
        if beats_global and reaches_trinity:
            status = "internal_gain_and_trinity_parity"
        elif beats_global:
            status = "internal_gain_but_trinity_gap"
        elif reaches_trinity:
            status = "trinity_parity_without_global_gain"
        else:
            status = "remaining_gap"
        output[name] = {
            "comparison_key": key,
            "global_slot_mean": global_mean,
            "candidate_mean": candidate_mean,
            "trinity_mean": trinity_mean,
            "beats_global_slot": beats_global,
            "reaches_trinity": reaches_trinity,
            "status": status,
        }
    return output


__all__ = [
    "ALPHA",
    "BETA",
    "COMPONENTS",
    "GROUPS",
    "SEEDS",
    "averaged_rows",
    "candidate_rows",
    "canonical_row",
    "component_report",
    "cost_decomposition",
    "headline_audit",
    "paired_relative",
    "read",
    "seed_average",
    "soft_contributions",
    "survival_report",
    "trinity_seed_rows",
    "vrel_decomposition",
]

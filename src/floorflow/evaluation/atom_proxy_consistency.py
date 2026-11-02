"""Audit whether cheap atom proxies preserve canonical metric decisions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

SCHEMA_VERSION = "atom-proxy-consistency-v1"


@dataclass(frozen=True)
class ConsistencyThresholds:
    min_checkpoints_per_seed: int = 6
    spearman_rho_min: float = 0.8
    pearson_rho_min: float = 0.7
    top_checkpoint_agreement_min: float = 0.9
    paired_sign_agreement_min: float = 0.8
    false_pass_rate_max: float = 0.05
    false_fail_rate_max: float = 0.1
    flat_proxy_epsilon: float = 1.0e-9

    @classmethod
    def from_contract(cls, contract: Mapping) -> ConsistencyThresholds:
        return cls(
            min_checkpoints_per_seed=int(
                contract.get("checkpoint_snapshots_per_seed_min", 6)
            ),
            spearman_rho_min=float(contract.get("spearman_rho_min", 0.8)),
            pearson_rho_min=float(
                contract.get("pearson_rho_secondary_min", 0.7)
            ),
            top_checkpoint_agreement_min=float(
                contract.get("top_checkpoint_decision_agreement_min", 0.9)
            ),
            paired_sign_agreement_min=float(
                contract.get("paired_sign_agreement_min", 0.8)
            ),
            false_pass_rate_max=float(
                contract.get("false_pass_rate_max", 0.05)
            ),
            false_fail_rate_max=float(
                contract.get("false_fail_rate_max", 0.1)
            ),
        )


@dataclass(frozen=True)
class Measurement:
    id: str
    atom_id: str
    proxy_metric: str
    canonical_metric: str
    proxy_direction: str
    canonical_direction: str
    baseline_checkpoint: str
    target_used_at_inference: bool
    rows: tuple[dict[str, Any], ...]
    sensitivity: tuple[dict[str, Any], ...]
    proxy_transform: str = "identity"
    canonical_transform: str = "identity"
    proxy_tolerance: float = 0.001
    canonical_tolerance: float = 0.001


def load_measurement_payloads(paths: Sequence[str | Path]) -> list[Measurement]:
    """Load one or more measurement payloads with a fail-closed schema."""

    measurements: list[Measurement] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("measurements", [payload])
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path} has no measurements")
        for raw in rows:
            measurement = _parse_measurement(raw)
            if measurement.id in seen:
                raise ValueError(f"duplicate measurement id: {measurement.id}")
            seen.add(measurement.id)
            measurements.append(measurement)
    return measurements


def _parse_measurement(raw: Mapping) -> Measurement:
    required = (
        "id",
        "atom_id",
        "proxy_metric",
        "canonical_metric",
        "rows",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"measurement missing fields: {missing}")
    rows = raw["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{raw['id']}: rows must be non-empty")
    parsed_rows = []
    seen: set[tuple[int, str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"{raw['id']}: measurement rows must be objects")
        parsed = {
            "seed": int(row["seed"]),
            "checkpoint": str(row["checkpoint"]),
            "case_id": str(row["case_id"]),
            "proxy": float(row["proxy"]),
            "canonical": float(row["canonical"]),
        }
        if not math.isfinite(parsed["proxy"]) or not math.isfinite(
            parsed["canonical"]
        ):
            raise ValueError(f"{raw['id']}: non-finite metric value")
        key = (
            parsed["seed"],
            parsed["checkpoint"],
            parsed["case_id"],
        )
        if key in seen:
            raise ValueError(f"{raw['id']}: duplicate row {key}")
        seen.add(key)
        parsed_rows.append(parsed)
    checkpoints = sorted({row["checkpoint"] for row in parsed_rows})
    baseline = str(raw.get("baseline_checkpoint", checkpoints[0]))
    if baseline not in checkpoints:
        raise ValueError(f"{raw['id']}: unknown baseline checkpoint {baseline}")
    target_used = bool(raw.get("target_used_at_inference", False))
    return Measurement(
        id=str(raw["id"]),
        atom_id=str(raw["atom_id"]),
        proxy_metric=str(raw["proxy_metric"]),
        canonical_metric=str(raw["canonical_metric"]),
        proxy_direction=str(raw.get("proxy_direction", "maximize")),
        canonical_direction=str(
            raw.get("canonical_direction", "maximize")
        ),
        baseline_checkpoint=baseline,
        target_used_at_inference=target_used,
        rows=tuple(parsed_rows),
        sensitivity=tuple(raw.get("sensitivity", [])),
        proxy_transform=str(raw.get("proxy_transform", "identity")),
        canonical_transform=str(
            raw.get("canonical_transform", "identity")
        ),
        proxy_tolerance=float(raw.get("proxy_tolerance", 0.001)),
        canonical_tolerance=float(raw.get("canonical_tolerance", 0.001)),
    )


def _adjusted(
    value: float,
    direction: str,
    transform: str,
) -> float:
    result = abs(value) if transform == "absolute" else value
    return -result if direction == "minimize" else result


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _correlations(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    if left.size < 2:
        return {"spearman": None, "pearson": None}
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return {"spearman": None, "pearson": None}
    return {
        "spearman": float(spearmanr(left, right).statistic),
        "pearson": float(pearsonr(left, right).statistic),
    }


def _case_values(
    measurement: Measurement,
) -> dict[tuple[int, str], dict[str, tuple[float, float]]]:
    result: dict[tuple[int, str], dict[str, tuple[float, float]]] = {}
    for row in measurement.rows:
        key = (int(row["seed"]), str(row["checkpoint"]))
        result.setdefault(key, {})[str(row["case_id"])] = (
            _adjusted(
                float(row["proxy"]),
                measurement.proxy_direction,
                measurement.proxy_transform,
            ),
            _adjusted(
                float(row["canonical"]),
                measurement.canonical_direction,
                measurement.canonical_transform,
            ),
        )
    return result


def _audit_attacks(measurement: Measurement) -> dict[str, object]:
    passed = 0
    considered = 0
    rows = []
    for raw in measurement.sensitivity:
        proxy_base = _adjusted(
            float(raw["proxy_baseline"]),
            measurement.proxy_direction,
            measurement.proxy_transform,
        )
        proxy_attack = _adjusted(
            float(raw["proxy_attack"]),
            measurement.proxy_direction,
            measurement.proxy_transform,
        )
        canonical_base = _adjusted(
            float(raw["canonical_baseline"]),
            measurement.canonical_direction,
            measurement.canonical_transform,
        )
        canonical_attack = _adjusted(
            float(raw["canonical_attack"]),
            measurement.canonical_direction,
            measurement.canonical_transform,
        )
        proxy_sensitive = proxy_attack < proxy_base
        canonical_sensitive = canonical_attack < canonical_base
        considered += 1
        passed += int(proxy_sensitive and canonical_sensitive)
        rows.append(
            {
                "case_id": raw.get("case_id"),
                "attack": raw.get("attack"),
                "proxy_sensitive": proxy_sensitive,
                "canonical_sensitive": canonical_sensitive,
                "passed": proxy_sensitive and canonical_sensitive,
            }
        )
    return {
        "cases": considered,
        "passed": passed,
        "rate": passed / considered if considered else None,
        "status": "complete" if considered else "missing",
        "rows": rows,
    }


def evaluate_measurement(
    measurement: Measurement,
    thresholds: ConsistencyThresholds,
) -> dict[str, object]:
    """Evaluate one proxy/canonical pair across checkpoints and seeds."""

    grouped = _case_values(measurement)
    seeds = sorted({seed for seed, _checkpoint in grouped})
    checkpoints = sorted({checkpoint for _seed, checkpoint in grouped})
    checkpoint_results = []
    per_seed = {}
    same_cases_by_seed = {}
    all_case_sets = []
    for seed in seeds:
        seed_groups = {
            checkpoint: cases
            for (row_seed, checkpoint), cases in grouped.items()
            if row_seed == seed
        }
        case_sets = [frozenset(cases) for cases in seed_groups.values()]
        all_case_sets.extend(case_sets)
        same_cases_by_seed[str(seed)] = (
            bool(case_sets)
            and all(case_set == case_sets[0] for case_set in case_sets)
        )
        means = {
            checkpoint: (
                float(np.mean([value[0] for value in cases.values()])),
                float(np.mean([value[1] for value in cases.values()])),
            )
            for checkpoint, cases in seed_groups.items()
        }
        for checkpoint in sorted(seed_groups):
            values = np.asarray(list(seed_groups[checkpoint].values()))
            result = {
                "seed": seed,
                "checkpoint": checkpoint,
                "cases": int(values.shape[0]),
                "proxy_mean": float(values[:, 0].mean()),
                "canonical_mean": float(values[:, 1].mean()),
                **_correlations(values[:, 0], values[:, 1]),
            }
            checkpoint_results.append(result)
        proxy_max = max(value[0] for value in means.values())
        canonical_max = max(value[1] for value in means.values())
        proxy_best = {
            name
            for name, value in means.items()
            if value[0] >= proxy_max - measurement.proxy_tolerance
        }
        canonical_best = {
            name
            for name, value in means.items()
            if value[1] >= canonical_max - measurement.canonical_tolerance
        }
        baseline_cases = seed_groups.get(measurement.baseline_checkpoint, {})
        paired_proxy = []
        paired_canonical = []
        for checkpoint, cases in seed_groups.items():
            if checkpoint == measurement.baseline_checkpoint:
                continue
            shared = sorted(set(cases) & set(baseline_cases))
            paired_proxy.extend(
                cases[case][0] - baseline_cases[case][0] for case in shared
            )
            paired_canonical.extend(
                cases[case][1] - baseline_cases[case][1] for case in shared
            )
        proxy_changes = np.asarray(paired_proxy)
        canonical_changes = np.asarray(paired_canonical)
        active = (
            (np.abs(proxy_changes) > measurement.proxy_tolerance)
            & (
                np.abs(canonical_changes)
                > measurement.canonical_tolerance
            )
        )
        sign_agreement = (
            float(
                (
                    np.sign(proxy_changes[active])
                    == np.sign(canonical_changes[active])
                ).mean()
            )
            if active.any()
            else None
        )
        false_pass = 0
        false_fail = 0
        comparison_count = 0
        for checkpoint, (proxy_mean, canonical_mean) in means.items():
            if checkpoint == measurement.baseline_checkpoint:
                continue
            proxy_delta = proxy_mean - means[
                measurement.baseline_checkpoint
            ][0]
            canonical_delta = canonical_mean - means[
                measurement.baseline_checkpoint
            ][1]
            comparison_count += 1
            false_pass += int(
                proxy_delta >= -measurement.proxy_tolerance
                and canonical_delta < -measurement.canonical_tolerance
            )
            false_fail += int(
                proxy_delta < -measurement.proxy_tolerance
                and canonical_delta >= -measurement.canonical_tolerance
            )
        proxy_values = [value[0] for value in means.values()]
        canonical_values = [value[1] for value in means.values()]
        per_seed[str(seed)] = {
            "checkpoints": len(means),
            "same_cases": same_cases_by_seed[str(seed)],
            "proxy_best_checkpoints": sorted(proxy_best),
            "canonical_best_checkpoints": sorted(canonical_best),
            "top_checkpoint_agreement": bool(proxy_best & canonical_best),
            "paired_sign_agreement": sign_agreement,
            "false_pass_rate": (
                false_pass / comparison_count if comparison_count else None
            ),
            "false_fail_rate": (
                false_fail / comparison_count if comparison_count else None
            ),
            "proxy_flat": (
                max(proxy_values) - min(proxy_values)
                <= thresholds.flat_proxy_epsilon
            ),
            "canonical_flat": (
                max(canonical_values) - min(canonical_values)
                <= thresholds.flat_proxy_epsilon
            ),
        }
    spearman_values = [
        float(row["spearman"])
        for row in checkpoint_results
        if row["spearman"] is not None
    ]
    pearson_values = [
        float(row["pearson"])
        for row in checkpoint_results
        if row["pearson"] is not None
    ]
    sign_values = [
        float(row["paired_sign_agreement"])
        for row in per_seed.values()
        if row["paired_sign_agreement"] is not None
    ]
    false_pass_values = [
        float(row["false_pass_rate"])
        for row in per_seed.values()
        if row["false_pass_rate"] is not None
    ]
    false_fail_values = [
        float(row["false_fail_rate"])
        for row in per_seed.values()
        if row["false_fail_rate"] is not None
    ]
    top_agreement = _mean(
        [
            float(row["top_checkpoint_agreement"])
            for row in per_seed.values()
        ]
    )
    attacks = _audit_attacks(measurement)
    same_cases_across_seeds = bool(all_case_sets) and all(
        case_set == all_case_sets[0]
        for case_set in all_case_sets[1:]
    )
    same_cases_all_seeds = (
        bool(same_cases_by_seed)
        and all(same_cases_by_seed.values())
        and same_cases_across_seeds
    )
    summary = {
        "spearman_rho": _mean(spearman_values),
        "pearson_rho": _mean(pearson_values),
        "top_checkpoint_agreement": top_agreement,
        "paired_sign_agreement": _mean(sign_values),
        "false_pass_rate": _mean(false_pass_values),
        "false_fail_rate": _mean(false_fail_values),
        "minimum_checkpoints_per_seed": min(
            (row["checkpoints"] for row in per_seed.values()),
            default=0,
        ),
        "same_cases_all_seeds": same_cases_all_seeds,
        "same_cases_across_seeds": same_cases_across_seeds,
        "flat_proxy": any(row["proxy_flat"] for row in per_seed.values()),
        "sensitivity_rate": attacks["rate"],
    }
    checks = {
        "target_free": not measurement.target_used_at_inference,
        "checkpoint_coverage": (
            summary["minimum_checkpoints_per_seed"]
            >= thresholds.min_checkpoints_per_seed
        ),
        "same_cases": summary["same_cases_all_seeds"],
        "spearman": (
            summary["spearman_rho"] is not None
            and summary["spearman_rho"] >= thresholds.spearman_rho_min
        ),
        "pearson": (
            summary["pearson_rho"] is not None
            and summary["pearson_rho"] >= thresholds.pearson_rho_min
        ),
        "top_checkpoint": (
            summary["top_checkpoint_agreement"] is not None
            and summary["top_checkpoint_agreement"]
            >= thresholds.top_checkpoint_agreement_min
        ),
        "paired_sign": (
            summary["paired_sign_agreement"] is not None
            and summary["paired_sign_agreement"]
            >= thresholds.paired_sign_agreement_min
        ),
        "false_pass": (
            summary["false_pass_rate"] is not None
            and summary["false_pass_rate"] <= thresholds.false_pass_rate_max
        ),
        "false_fail": (
            summary["false_fail_rate"] is not None
            and summary["false_fail_rate"] <= thresholds.false_fail_rate_max
        ),
        "not_flat": not summary["flat_proxy"],
        "sensitivity": (
            summary["sensitivity_rate"] is not None
            and summary["sensitivity_rate"] >= 1.0
        ),
    }
    return {
        "measurement_id": measurement.id,
        "atom_id": measurement.atom_id,
        "proxy_metric": measurement.proxy_metric,
        "canonical_metric": measurement.canonical_metric,
        "seeds": seeds,
        "checkpoints": checkpoints,
        "summary": summary,
        "checks": checks,
        "passed": all(checks.values()),
        "per_seed": per_seed,
        "checkpoint_results": checkpoint_results,
        "sensitivity": attacks,
    }


def build_consistency_report(
    measurements: Sequence[Measurement],
    thresholds: ConsistencyThresholds,
) -> dict[str, object]:
    """Evaluate all supplied measurements and preserve missing-data states."""

    results = [
        evaluate_measurement(measurement, thresholds)
        for measurement in measurements
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "thresholds": {
            "min_checkpoints_per_seed": thresholds.min_checkpoints_per_seed,
            "spearman_rho_min": thresholds.spearman_rho_min,
            "pearson_rho_min": thresholds.pearson_rho_min,
            "top_checkpoint_agreement_min": (
                thresholds.top_checkpoint_agreement_min
            ),
            "paired_sign_agreement_min": (
                thresholds.paired_sign_agreement_min
            ),
            "false_pass_rate_max": thresholds.false_pass_rate_max,
            "false_fail_rate_max": thresholds.false_fail_rate_max,
        },
        "summary": {
            "measurements": len(results),
            "passed": sum(bool(row["passed"]) for row in results),
            "failed": sum(not bool(row["passed"]) for row in results),
        },
        "measurements": results,
    }


__all__ = [
    "SCHEMA_VERSION",
    "ConsistencyThresholds",
    "Measurement",
    "build_consistency_report",
    "evaluate_measurement",
    "load_measurement_payloads",
]

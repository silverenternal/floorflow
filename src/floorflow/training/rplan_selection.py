"""Gate-balanced checkpoint selection for RPLAN generation tracks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _higher(value: float, target: float) -> float:
    return min(1.25, max(-1.25, value / target))


def _lower(value: float, target: float) -> float:
    return min(1.25, max(-1.25, 1.0 - value / target))


def rplan_selection_score(
    metrics: Mapping[str, Any],
    *,
    token_enabled: bool,
    freeze_non_stage: bool = False,
) -> float:
    """Average gate progress over every available RPLAN quality atom."""

    progress: list[float] = []
    if freeze_non_stage:
        progress.extend(
            [
                _lower(float(metrics.get("wall_total_loss", 0.0)), 0.7),
                _lower(float(metrics.get("opening_total_loss", 0.0)), 0.3),
            ]
        )
    else:
        progress.extend(
            [
                _higher(
                    float(metrics.get("matched_geometric_iou", 0.0)),
                    0.45,
                ),
                _higher(
                    float(metrics.get("instance_f1_at_threshold", 0.0)),
                    0.50,
                ),
                _higher(
                    float(metrics.get("matched_type_accuracy", 0.0)),
                    0.75,
                ),
                _lower(
                    float(
                        metrics.get(
                            "room_count_mae",
                            metrics.get("count_abs_error", 2.8),
                        )
                    ),
                    2.8,
                ),
            ]
        )
        progress.extend(
            [
                _lower(float(metrics.get("wall_total_loss", 0.0)), 0.7),
                _lower(float(metrics.get("opening_total_loss", 0.0)), 0.3),
                _lower(float(metrics.get("room_adjacency_bce", 0.0)), 0.3),
            ]
        )
    contract = metrics.get("contract", {})
    if isinstance(contract, Mapping) and "contract_valid" in contract:
        progress.append(float(bool(contract["contract_valid"])))
    if token_enabled:
        token_values = (
            ("token_accuracy", 0.95, True),
            ("token_macro_f1", 0.90, True),
            ("canonical_token_ce", 3.0, False),
            ("canonical_token_grammar_violation_rate", 0.01, False),
            ("canonical_token_structure_loss", 0.1, False),
        )
        for name, target, higher_is_better in token_values:
            if name not in metrics:
                continue
            value = float(metrics[name])
            progress.append(
                _higher(value, target)
                if higher_is_better
                else _lower(value, target)
            )
        generation = metrics.get("token_generation", {})
        if isinstance(generation, Mapping):
            for name in (
                "token_generation_completed",
                "token_generation_within_efficiency_budget",
            ):
                if name in generation:
                    progress.append(float(bool(generation[name])))
        consumer = metrics.get("token_consumer", {})
        if isinstance(consumer, Mapping):
            progress.extend(
                float(bool(consumer.get(name, 0.0)))
                for name in (
                    "token_consumer_generated_contract_valid",
                    "token_consumer_generated_functional_watertight",
                )
            )
            for name in (
                "token_consumer_generated_validity_regression",
                "token_consumer_generated_watertight_regression",
            ):
                if name in consumer:
                    progress.append(_lower(float(consumer[name]), 0.05))
    if not progress:
        raise ValueError("RPLAN selection has no comparable metrics")
    return sum(progress) / len(progress)


__all__ = ["rplan_selection_score"]

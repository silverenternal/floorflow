"""Static capacity contract between RPLAN targets and generator outputs."""

from __future__ import annotations

from collections.abc import Mapping

from floorflow.data.rplan_boundary_adapter import RPlanTensorLimits


def validate_rplan_model_limits(
    limits: RPlanTensorLimits,
    model_config: Mapping[str, object],
) -> None:
    """Reject target capacities that exceed the configured generator."""

    requirements = (
        ("identity_queries", limits.max_rooms),
        ("max_walls", limits.max_walls),
        ("max_openings", limits.max_openings),
        ("polygon_points", limits.room_points),
        ("max_boundary_points", limits.max_boundary_points),
    )
    violations = []
    for key, required in requirements:
        available = int(model_config.get(key, 0))
        if available < int(required):
            violations.append(f"{key}={available} < target {required}")
    if violations:
        raise ValueError(
            "RPLAN model capacity is smaller than tensor targets: "
            + "; ".join(violations)
        )


__all__ = ["validate_rplan_model_limits"]

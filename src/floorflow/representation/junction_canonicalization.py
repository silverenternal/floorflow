"""Stable fixed-point junction derivation for canonical floorplans."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from floorflow.evaluation.canonical_floorplan import (
    CanonicalJunction,
    CanonicalWall,
)

if TYPE_CHECKING:
    from floorflow.representation.unified_floorplan_tokens import (
        CoordinateQuantizer,
    )


def derive_junctions(
    walls: tuple[CanonicalWall, ...],
    quantizer: CoordinateQuantizer,
    tolerance: float,
) -> tuple[CanonicalJunction, ...]:
    """Cluster wall endpoints and assign IDs from quantized coordinates."""

    clusters: list[dict[str, object]] = []
    for wall in walls:
        for point in (wall.start, wall.end):
            for cluster in clusters:
                center = cluster["center"]
                assert isinstance(center, tuple)
                if math.dist(point, center) <= tolerance:
                    points = cluster["points"]
                    wall_ids = cluster["wall_ids"]
                    assert isinstance(points, list)
                    assert isinstance(wall_ids, set)
                    points.append(point)
                    wall_ids.add(wall.id)
                    cluster["center"] = (
                        sum(item[0] for item in points) / len(points),
                        sum(item[1] for item in points) / len(points),
                    )
                    break
            else:
                clusters.append(
                    {"center": point, "points": [point], "wall_ids": {wall.id}}
                )

    def quantized_center(cluster: dict[str, object]) -> tuple[float, float]:
        center = cluster["center"]
        assert isinstance(center, tuple)
        return (
            quantizer.dequantize(quantizer.quantize(center[0])),
            quantizer.dequantize(quantizer.quantize(center[1])),
        )

    ordered = sorted(clusters, key=quantized_center)
    return tuple(
        CanonicalJunction(
            id=f"junction_{index:04d}",
            point=quantized_center(cluster),
            degree=len(cluster["wall_ids"]),
        )
        for index, cluster in enumerate(ordered, start=1)
    )


__all__ = ["derive_junctions"]

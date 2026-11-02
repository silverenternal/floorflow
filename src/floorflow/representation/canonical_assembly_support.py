"""Input normalization helpers for deterministic canonical assembly."""

from __future__ import annotations

from collections.abc import Sequence

from floorflow.evaluation.canonical_floorplan import (
    CanonicalJunction,
    CanonicalOpening,
)
from floorflow.representation.unified_floorplan_tokens import (
    CoordinateQuantizer,
)


def adaptive_assembly_quantizer(
    width: int,
    height: int,
    *,
    reference_scale: int = 16,
    reference_resolution: int = 384,
) -> CoordinateQuantizer:
    scale = max(
        4,
        round(
            float(reference_scale)
            * min(int(width), int(height))
            / max(1, int(reference_resolution))
        ),
    )
    return CoordinateQuantizer(scale=scale)


def canonical_openings_from_rows(
    rows: Sequence[dict] | None,
    wall_ids: Sequence[str],
) -> tuple[CanonicalOpening, ...]:
    if not rows:
        return ()
    openings = []
    for index, row in enumerate(rows, start=1):
        raw_wall_id = row.get("wall_id")
        if isinstance(raw_wall_id, str):
            wall_id = raw_wall_id
        else:
            wall_index = int(
                row.get("wall_index", raw_wall_id or 0)
            )
            if not 0 <= wall_index < len(wall_ids):
                continue
            wall_id = wall_ids[wall_index]
        openings.append(
            CanonicalOpening(
                id=str(row.get("id") or f"opening_{index:04d}"),
                label=str(row["label"]),
                center=tuple(map(float, row["center"])),
                width=float(row["width"]),
                wall_id=wall_id,
                confidence=float(row.get("confidence", 1.0)),
            )
        )
    return tuple(openings)


def canonical_junctions_from_rows(
    rows: Sequence[dict] | None,
) -> tuple[CanonicalJunction, ...]:
    return tuple(
        CanonicalJunction(
            id=str(row.get("id") or f"junction_{index:04d}"),
            point=tuple(map(float, row["point"])),
            degree=int(row["degree"]),
        )
        for index, row in enumerate(rows or (), start=1)
    )


__all__ = [
    "adaptive_assembly_quantizer",
    "canonical_junctions_from_rows",
    "canonical_openings_from_rows",
]

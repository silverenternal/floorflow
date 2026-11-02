"""Select per-style conditioning strengths on validation reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping


def select_style_gates(
    reports: Iterable[Mapping[str, object]],
    *,
    minimum_coverage: float,
) -> dict[str, dict[str, float]]:
    """Maximize quality subject to nonzero density and baseline coverage."""

    candidates: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for report in reports:
        style_id = report.get("style_id")
        if style_id is None:
            continue
        candidates[int(style_id)].append(report)
    selected = {}
    for style_id, rows in sorted(candidates.items()):
        admissible = [
            row
            for row in rows
            if float(row["distribution"]["coverage"])
            >= float(minimum_coverage)
            and float(row["distribution"]["density"]) > 0.0
        ]
        pool = admissible or list(rows)
        best = max(
            pool,
            key=lambda row: (
                float(row["sample_quality"]["mean"]),
                float(row["distribution"]["coverage"]),
            ),
        )
        selected[str(style_id)] = {
            "gate": float(best["style_gate"]),
            "quality": float(best["sample_quality"]["mean"]),
            "coverage": float(best["distribution"]["coverage"]),
            "density": float(best["distribution"]["density"]),
            "coverage_gate_satisfied": bool(admissible),
        }
    return selected


__all__ = ["select_style_gates"]

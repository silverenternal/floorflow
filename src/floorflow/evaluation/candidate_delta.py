"""Safe metric deltas across candidates with unequal metric coverage."""

from __future__ import annotations


def shared_variant_delta(
    summary: dict[str, dict[str, float]],
    name: str,
) -> dict[str, float] | None:
    """Compare only metrics present in both the reference and candidate."""

    if name not in summary or "raw" not in summary:
        return None
    shared = summary[name].keys() & summary["raw"].keys()
    return {
        key: summary[name][key] - summary["raw"][key]
        for key in shared
    }


__all__ = ["shared_variant_delta"]

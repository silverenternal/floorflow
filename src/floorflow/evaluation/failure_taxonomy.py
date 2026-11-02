"""Rule-based room failure taxonomy from error-decomposition rows."""

from __future__ import annotations


def classify_room_failure(
    row: dict,
    *,
    fragmentation_threshold: float,
    coverage_threshold: float = 0.85,
) -> str:
    allocation = row["allocation"]
    if allocation["missing_room_fraction"] >= max(
        allocation["semantic_mismatch_fraction"],
        allocation["spurious_room_fraction"],
        0.25,
    ):
        return "missing_room"
    if allocation["semantic_mismatch_fraction"] >= max(
        allocation["spurious_room_fraction"],
        0.25,
    ):
        return "semantic_confusion"
    if allocation["spurious_room_fraction"] >= 0.25:
        return "spurious_room"
    if float(row["support_fragmentation_gap"]) >= fragmentation_threshold:
        return "instance_fragmentation"
    if float(row.get("support_occupancy_iou", 1.0)) < coverage_threshold:
        return "boundary_mismatch"
    return "mixed_or_local"


def summarise_failure_taxonomy(
    rows: list[dict],
    *,
    fragmentation_threshold: float,
) -> dict:
    categories: dict[str, list[dict]] = {}
    for row in rows:
        name = classify_room_failure(
            row,
            fragmentation_threshold=fragmentation_threshold,
        )
        categories.setdefault(name, []).append(row)
    total = max(1, len(rows))
    return {
        "case_count": len(rows),
        "categories": {
            name: {
                "count": len(items),
                "fraction": len(items) / total,
                "worst_cases": sorted(
                    items,
                    key=lambda item: item["support_macro_iou"],
                )[:10],
            }
            for name, items in sorted(
                categories.items(),
                key=lambda pair: len(pair[1]),
                reverse=True,
            )
        },
    }

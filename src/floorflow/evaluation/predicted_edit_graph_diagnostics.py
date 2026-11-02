"""Paired graph-effect diagnostics for predicted floorplan edits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from floorflow.evaluation.paired_bootstrap import (
    paired_delta,
)


def predicted_edit_graph_effect(
    rows: Sequence[dict[str, Any]],
    *,
    iterations: int = 1000,
) -> dict[str, dict[str, Any]]:
    """Compare post-edit and pre-edit refined graphs on identical executed cases."""

    executed = [row for row in rows if not row["abstained"]]
    current = [
        {"index": index, **row["graph"]["refined"]}
        for index, row in enumerate(executed)
    ]
    before = [
        {"index": index, **row["pre_edit_graph"]["refined"]}
        for index, row in enumerate(executed)
    ]
    if not current:
        raise ValueError("graph-effect diagnostics require executed edits")
    return {
        "node_f1": paired_delta(
            current,
            before,
            "node_f1",
            iterations=iterations,
        ),
        "edge_overlap": paired_delta(
            current,
            before,
            "edge_overlap",
            iterations=iterations,
        ),
        "ged": paired_delta(
            current,
            before,
            "ged",
            iterations=iterations,
            lower_is_better=True,
        ),
    }


__all__ = ["predicted_edit_graph_effect"]

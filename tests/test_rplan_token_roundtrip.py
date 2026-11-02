from __future__ import annotations

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    rplan_taxonomy,
)
from floorflow.evaluation.rplan_token_roundtrip import (
    audit_rplan_token_roundtrip,
)


def _row() -> dict:
    return {
        "name": "roundtrip",
        "boundary": [[0, 0], [0, 100], [100, 100], [100, 0]],
        "rooms": [
            {
                "id": 0,
                "category": 0,
                "location": [25, 50],
                "r_boundary": [[0, 0], [50, 0], [50, 100], [0, 100]],
            }
        ],
        "adjacencies": [],
    }


def test_rplan_token_roundtrip_preserves_structure_within_half_step() -> None:
    taxonomy = rplan_taxonomy()
    dataset = RPlanBoundaryDataset(
        "unused.json",
        taxonomy,
        rows=[_row()],
    )
    result = audit_rplan_token_roundtrip(dataset._plan(0), taxonomy)
    assert result["passed"]
    assert result["structural_counts_match"]
    assert result["token_text_exact"]
    assert result["max_quantization_error"] <= result["half_step"] + 1.0e-6

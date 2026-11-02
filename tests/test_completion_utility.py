from __future__ import annotations

from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.completion_utility import completion_utility_metrics


def test_completion_utility_wires_edit_and_intent_metrics(monkeypatch) -> None:
    taxonomy = FloorplanTaxonomy.read(
        Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json")
    )
    calls = []

    def fake_fpeval(prediction, target, *args, **kwargs):
        calls.append((prediction.copy(), target.copy()))
        edit_cost = float(len(calls) * 10)
        return {"summary": {"0.05": {"edit_cost": edit_cost}}}

    monkeypatch.setattr(
        "floorflow.evaluation.completion_utility.fpeval_raster_metrics",
        fake_fpeval,
    )
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[:, :4] = taxonomy.room_ids[0]
    labels[:, 4:] = taxonomy.room_ids[1]
    result = completion_utility_metrics(
        labels,
        labels,
        labels,
        taxonomy,
        output_edit_cost=5.0,
        output_edit_cost_per_gt_wall=1.0,
    )
    assert len(calls) == 2
    assert result["fpeval_edit_cost"] == 5.0
    assert result["control_edit_cost_to_target"] == 10.0
    assert result["edit_cost_reduction"] == 5.0
    assert result["intent_edit_cost"] == 20.0
    assert result["control_room_pixel_preservation"] == 1.0

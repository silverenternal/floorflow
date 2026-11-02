import math

from floorflow.evaluation.candidate_delta import shared_variant_delta


def test_shared_variant_delta_ignores_unequal_metric_sets() -> None:
    summary = {
        "raw": {"room": 0.2, "object": 0.1},
        "hybrid": {"room": 0.25, "polygon": 0.3},
    }
    delta = shared_variant_delta(summary, "hybrid")
    assert delta is not None
    assert math.isclose(delta["room"], 0.05)
    assert shared_variant_delta(summary, "missing") is None

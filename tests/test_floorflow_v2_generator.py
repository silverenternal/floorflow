from __future__ import annotations

from floorflow.data.rplan_boundary_adapter import rplan_taxonomy
from floorflow.model.floorflow_v2_generator import build_floorflow_v2_generator


def test_floorflow_v2_is_parameter_matched_and_architecture_first() -> None:
    model = build_floorflow_v2_generator(rplan_taxonomy())
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 0.9 * 9_636_614 <= parameters <= 1.1 * 9_636_614
    assert model.config.use_boundary_edge_graph
    assert model.config.enable_point_validity
    assert model.config.encoder_architecture == "compact"
    assert not model.config.enable_query_quality

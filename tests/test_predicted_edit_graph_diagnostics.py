from floorflow.evaluation.predicted_edit_graph_diagnostics import (
    predicted_edit_graph_effect,
)


def _graph(ged: float, node_f1: float, edge: float) -> dict:
    return {
        "ged": ged,
        "node_f1": node_f1,
        "edge_overlap": edge,
    }


def test_predicted_edit_graph_effect_isolates_executed_before_after() -> None:
    rows = [
        {
            "abstained": False,
            "pre_edit_graph": {"refined": _graph(10.0, 0.2, 0.5)},
            "graph": {"refined": _graph(8.0, 0.4, 0.7)},
        },
        {
            "abstained": False,
            "pre_edit_graph": {"refined": _graph(12.0, 0.3, 0.6)},
            "graph": {"refined": _graph(9.0, 0.5, 0.8)},
        },
        {
            "abstained": True,
            "pre_edit_graph": {"refined": _graph(0.0, 1.0, 1.0)},
            "graph": {"refined": _graph(0.0, 1.0, 1.0)},
        },
    ]
    effect = predicted_edit_graph_effect(rows, iterations=50)
    assert effect["node_f1"]["mean_delta"] == 0.2
    assert effect["edge_overlap"]["mean_delta"] == 0.2
    assert effect["ged"]["mean_delta"] == -2.5
    assert effect["ged"]["improved"] == 2

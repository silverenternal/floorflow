from pathlib import Path

from floorflow.chip_flow.def_graph import parse_def_graph
from floorflow.chip_flow.ppa_external import graph_edges

ROOT = Path(__file__).resolve().parents[1]
CHIPBENCH = ROOT / ".data/chipbench_d_v1/data/ariane81"


def test_parse_chipbench_def_without_openroad() -> None:
    macros = [
        path
        for path in sorted((CHIPBENCH / "lef").glob("*.lef"))
        if not path.name.startswith("NangateOpenCellLibrary")
    ]
    graph = parse_def_graph(
        CHIPBENCH / "def/pre_place.def",
        macros,
    )
    assert len(graph.macros) == 81
    assert len(graph.ports) == 495
    assert graph.core[0] > 0 and graph.core[1] > 0
    assert all(
        macro["width_dbu"] > 0 and macro["height_dbu"] > 0
        for macro in graph.macros
    )
    edges = graph_edges(graph)
    assert edges["b2b"]
    assert edges["p2b"]
    assert edges["used_ports"]

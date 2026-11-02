from pathlib import Path

from floorflow.chip_flow.trinity_release import load_release


def test_pinned_trinity_release_loads_without_lightning() -> None:
    root = Path(__file__).resolve().parents[1]
    release = load_release(
        root / "third_party/Trinity/checkpoints/d192",
        "cpu",
    )
    parameters = sum(parameter.numel() for parameter in release.backbone.parameters())
    assert parameters == 5_856_400
    assert release.config["arch"]["output_kind"] == "x0"
    assert release.graph_pe.dim == 2

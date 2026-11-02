import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/chip_sota_inference_search_frontier_v1.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_inference_frontier_is_frozen_and_reuses_existing_manifests() -> None:
    config = load_config()
    assert config["status"].startswith("frozen_")
    assert config["training"]["steps"] == 0
    for path in config["manifests"].values():
        assert (ROOT / path).is_file(), path
    stages = {stage["id"]: stage for stage in config["stages"]}
    assert stages["S4-e-oracle-draw-ceiling"]["selection_mode"] == "oracle_diagnostic"
    assert stages["S4-f-deployable-draws"]["selection_mode"] == "deployable_selector"


def test_inference_frontier_preserves_budget_and_claim_boundaries() -> None:
    config = load_config()
    grids = config["grids"]
    assert grids["one_sample"]["nfe_curve"]["nfe"] == [1, 2, 4, 8, 16, 32]
    assert grids["one_sample"]["refiner_curve"]["refiner_steps"] == [
        0,
        50,
        100,
        200,
        400,
        800,
        1600,
        3200,
    ]
    assert grids["multi_draw"]["draws"] == [1, 4, 16, 48]
    assert config["claim_gates"]["required_guard"].startswith(
        "Legalized feasibility remains 1.0"
    )
    assert any(
        "target-derived oracle" in rule
        for rule in config["claim_gates"]["forbidden"]
    )

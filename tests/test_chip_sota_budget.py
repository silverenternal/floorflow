import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (ROOT / "configs/chip_sota_budget_ladder_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_sota_budget_audit_preserves_parameter_match() -> None:
    config = _config()
    ours = config["current_model"]["parameters"]
    d192 = config["anchors"]["trinity_d192"]["parameters"]

    assert ours == 5_836_447
    assert d192 == 5_856_400
    assert abs(ours / d192 - 1.0) < 0.01


def test_training_exposure_gap_is_twenty_times() -> None:
    config = _config()
    current = config["current_model"]
    anchor = config["anchors"]["trinity_d192"]

    current_exposures = current["training_steps"] * current["effective_batch"]
    anchor_exposures = anchor["training_steps"] * anchor["effective_batch"]
    assert current_exposures == 2_560_000
    assert anchor_exposures == 51_200_000
    assert anchor_exposures / current_exposures == 20.0


def test_size_ladder_is_monotonic_and_scoped() -> None:
    config = _config()
    ladder = config["parameter_ladder"]

    assert [row["parameters"] for row in ladder] == sorted(
        row["parameters"] for row in ladder
    )
    assert ladder[0]["parameters"] == 5_836_447
    assert ladder[-1]["parameters"] == 86_445_345
    assert config["training_compute_ladder"]["data_layer"].startswith(
        "frozen 20k-case subset"
    )


def test_budget_claims_are_explicitly_separated() -> None:
    config = _config()
    assert set(config["claim_types"]) == {
        "absolute_sota",
        "equal_training_sota",
        "equal_inference_sota",
        "equal_quality_efficiency",
    }
    assert "Oracle best-of-N is diagnostic only" in config["inference_ladder"][
        "selection_policy"
    ]


def test_adaptive_inference_has_no_target_leakage_and_counts_compute() -> None:
    config = _config()
    adaptive = config["adaptive_inference"]

    assert adaptive["snapshots"] == [400, 800, 1600, 3200]
    assert "official_cost" in adaptive["forbidden_inputs"]
    assert "case_identity" in adaptive["forbidden_inputs"]
    assert "scorer calls" in adaptive["compute_accounting"]


def test_sota_portfolio_starts_with_data_and_adaptive_inference() -> None:
    config = _config()
    routes = {row["id"]: row for row in config["route_portfolio"]}

    assert routes["B1-training-compute"]["priority"] == "P0"
    assert routes["B2-high-block-data-coverage"]["priority"] == "P0"
    assert routes["B3-adaptive-refiner-and-search"]["priority"] == "closed_negative"
    assert routes["B4-capacity-scaling"]["priority"] == "P1"


def test_range_training_ladder_matches_frozen_hashes() -> None:
    ladder = json.loads(
        (
            ROOT / "configs/chip_sota_training_range_compute_ladder_v1.json"
        ).read_text(encoding="utf-8")
    )
    subset = json.loads(
        (
            ROOT / "configs/chip_train_scaling_range20k_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert ladder["train_indices"] == "configs/chip_train_scaling_range20k_v1.json"
    assert ladder["train_indices_sha256"] == (
        "21562442d64ee0ce0c8eda44b6a05b8196b88c499fa40e5f33f41ad6b1d7e32e"
    )
    assert ladder["cache_metadata_sha256"] == (
        "d3101b5f2569ebf642e5ef7e6857d510d44ce32b3434b7ab105afcffc8c0c428"
    )
    assert subset["cache"] == (
        "outputs/cache/chip_floorset_200k_train_12k_dev_range_v1/train"
    )


def test_frozen_scaling_subset_contract() -> None:
    subset = json.loads(
        (ROOT / "configs/chip_train_scaling_20k_v1.json").read_text(
            encoding="utf-8"
        )
    )
    screen = json.loads(
        (ROOT / "configs/chip_train_screen_5k_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert subset["version"] == "chip-train-scaling-20k-v1"
    assert subset["case_count"] == 20_000
    assert len(subset["indices"]) == 20_000
    assert len(set(subset["indices"])) == 20_000
    assert subset["cache_metadata_sha256"] == screen["cache_metadata_sha256"]


def test_range_scaling_subset_covers_training_outlier() -> None:
    subset = json.loads(
        (
            ROOT / "configs/chip_train_scaling_range20k_v1.json"
        ).read_text(encoding="utf-8")
    )
    histogram = subset["block_count_histogram"]

    assert subset["version"] == "chip-train-scaling-range20k-v1"
    assert subset["case_count"] == 20_000
    assert min(map(int, histogram)) == 21
    assert max(map(int, histogram)) == 120
    assert histogram["120"] > 0


def test_refiner_budget_screen_spans_frozen_dev_range() -> None:
    manifest = json.loads(
        (
            ROOT / "outputs/reports/chip_sota_refiner_budget_screen8_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["cases"] == 8
    assert len(set(manifest["indices"])) == 8
    assert [row["block_count"] for row in manifest["rows"]] == [
        21,
        35,
        49,
        63,
        78,
        92,
        106,
        120,
    ]


def test_official8_budget_manifest_is_predeclared_and_canonical() -> None:
    manifest = json.loads(
        (
            ROOT / "outputs/reports/chip_sota_official8_manifest_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "frozen_before_outcome"
    assert manifest["indices"] == [0, 14, 28, 42, 57, 71, 85, 99]
    assert manifest["blocks"] == [21, 35, 49, 63, 78, 92, 106, 120]


def test_highblock8_manifest_is_predeclared_and_canonical() -> None:
    manifest = json.loads(
        (
            ROOT / "outputs/reports/chip_sota_official_highblock8_manifest_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "frozen_before_outcome"
    assert manifest["indices"] == [40, 48, 57, 65, 74, 82, 91, 99]
    assert manifest["blocks"] == [61, 69, 78, 86, 95, 103, 112, 120]


def test_highblock8_pair_retains_distribution_gap() -> None:
    report = json.loads(
        (
            ROOT / "outputs/reports/chip_sota_official_highblock8_pair_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "pilot_not_supported"
    assert report["relative_delta"]["legalized_hard_cost"] > 0.2
    assert report["relative_delta"]["legalized_v_rel"] > 1.3


def test_highblock_refiner1600_is_near_anchor_and_not_a_win() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/reports/chip_sota_official_highblock8_pair_refine1600_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "pilot_not_supported"
    assert report["relative_delta"]["legalized_hard_cost"] == pytest.approx(
        0.01664398906697717
    )
    assert report["relative_delta"]["legalized_v_rel"] == pytest.approx(
        0.02082387337217284
    )


def test_official8_budget_pair_retains_negative_result() -> None:
    report = json.loads(
        (ROOT / "outputs/reports/chip_sota_official8_pair_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "pilot_not_supported"
    assert report["paired"]["legalized_hard_cost"]["ci_low"] < 0.0
    assert report["paired"]["legalized_hard_cost"]["ci_high"] > 0.0
    assert report["paired"]["legalized_v_rel"]["ci_low"] < 0.0
    assert report["paired"]["legalized_v_rel"]["ci_high"] > 0.0

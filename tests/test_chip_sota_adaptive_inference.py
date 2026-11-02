import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _manifest_module():
    path = ROOT / "scripts/build_chip_sota_adaptive_manifests.py"
    spec = importlib.util.spec_from_file_location("adaptive_manifest_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adaptive_analysis_module():
    path = ROOT / "scripts/calibrate_chip_adaptive_refiner.py"
    spec = importlib.util.spec_from_file_location("adaptive_refiner_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adaptive_manifest_split_is_stratified_and_disjoint() -> None:
    manifest_module = _manifest_module()
    rows = [
        {"index": 10 + offset, "block_count": 21 + offset}
        for offset in range(32)
    ]
    strata = manifest_module.split_strata(rows, 8)

    assert len(strata) == 8
    assert all(len(stratum) == 4 for stratum in strata)
    assert len({row["index"] for stratum in strata for row in stratum}) == 32


def test_generated_manifests_exclude_prior_refiner_screen() -> None:
    manifest_dir = ROOT / "outputs/reports/chip_sota_adaptive_manifests_v1"
    calibration = json.loads((manifest_dir / "calibration8.json").read_text())
    validation = json.loads((manifest_dir / "validation8.json").read_text())
    prior = json.loads(
        (ROOT / "outputs/reports/chip_sota_refiner_budget_screen8_v1.json").read_text()
    )

    manifest_module = _manifest_module()
    calibration_indices = manifest_module.manifest_indices(calibration)
    validation_indices = manifest_module.manifest_indices(validation)
    assert calibration["role"] == "calibration"
    assert validation["role"] == "validation"
    assert len(calibration_indices) == len(validation_indices) == 8
    assert not calibration_indices & validation_indices
    assert not (calibration_indices | validation_indices) & manifest_module.manifest_indices(
        prior
    )


def test_adaptive_config_forbids_target_leakage() -> None:
    config = json.loads(
        (ROOT / "configs/chip_sota_adaptive_inference_v1.json").read_text()
    )

    assert config["snapshots"] == [400, 800, 1600, 3200]
    assert set(config["forbidden_runtime_inputs"]) == {
        "target boxes",
        "official cost",
        "V_rel",
        "case identity",
    }


def test_adaptive_manifest_hashes_match_frozen_config() -> None:
    config = json.loads(
        (ROOT / "configs/chip_sota_adaptive_inference_v1.json").read_text()
    )
    for role in ("calibration_manifest", "validation_manifest"):
        path = ROOT / config[role]["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == config[role]["sha256"]


def test_plateau_policy_uses_observable_energy_only() -> None:
    module = _adaptive_analysis_module()
    observables = {
        1: {
            400: {"total": 10.0},
            800: {"total": 9.0},
            1600: {"total": 8.99},
            3200: {"total": 8.98},
        },
        2: {
            400: {"total": 20.0},
            800: {"total": 18.0},
            1600: {"total": 17.98},
            3200: {"total": 17.97},
        },
    }
    policy = {
        "kind": "energy_plateau",
        "metric": "total",
        "threshold": 0.0015,
        "min_steps": 800,
    }

    assert module.apply_policy(observables, [400, 800, 1600, 3200], policy) == {
        1: 1600,
        2: 1600,
    }


def test_rank_policy_uses_only_the_observable_ranking() -> None:
    module = _adaptive_analysis_module()
    observables = {
        1: {
            400: {"total": 10.0},
            800: {"total": 8.0},
            1600: {"total": 9.0},
            3200: {"total": 7.0},
        },
        2: {
            400: {"total": 20.0},
            800: {"total": 17.0},
            1600: {"total": 18.0},
            3200: {"total": 16.0},
        },
    }

    selected = module.apply_policy(
        observables,
        [400, 800, 1600, 3200],
        {"kind": "observable_rank", "metric": "total", "direction": "min"},
    )
    assert selected == {1: 3200, 2: 3200}


def test_calibration_serializes_policy_and_never_uses_case_identity() -> None:
    module = _adaptive_analysis_module()
    observables = {
        index: {
            400: {"total": 10.0},
            800: {"total": 9.0},
            1600: {"total": 8.9},
            3200: {"total": 8.8},
        }
        for index in range(1, 9)
    }
    legalized = {
        index: {
            400: {
                "blocks": 40,
                "legalized_hard_cost": 1.3,
                "legalized_v_rel": 0.04,
                "legalized_feasible": True,
            },
            800: {
                "blocks": 40,
                "legalized_hard_cost": 1.2,
                "legalized_v_rel": 0.03,
                "legalized_feasible": True,
            },
            1600: {
                "blocks": 40,
                "legalized_hard_cost": 1.1,
                "legalized_v_rel": 0.02,
                "legalized_feasible": True,
            },
            3200: {
                "blocks": 40,
                "legalized_hard_cost": 1.11,
                "legalized_v_rel": 0.025,
                "legalized_feasible": True,
            },
        }
        for index in range(1, 9)
    }
    result = module.calibrate(
        observables,
        legalized,
        [400, 800, 1600, 3200],
        control_steps=1600,
        thresholds=[0.001],
        metrics=["total"],
        min_steps=800,
        tolerance=0.005,
    )

    assert result["status"] == "policy_frozen"
    assert result["policy"]["kind"] in {"fixed", "energy_plateau"}
    assert result["selected"]["cases"] == 8

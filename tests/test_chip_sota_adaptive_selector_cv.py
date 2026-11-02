import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/analyze_chip_adaptive_selector_cv.py"
    spec = importlib.util.spec_from_file_location("adaptive_selector_cv", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ridge_predictor_returns_one_prediction_per_row() -> None:
    module = _module()
    x_train = np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_train = np.asarray([0.0, 1.0, 0.5])
    x_test = np.asarray([[0.5, 0.5], [2.0, 2.0]])

    prediction = module.fit_predict(x_train, y_train, x_test, 0.1)

    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()


def test_fixed_summary_uses_block_weighted_cost() -> None:
    module = _module()
    cases = {
        1: {
            400: {
                "observable": {"refiner_steps": 400},
                "legalized": {
                    "blocks": 20,
                    "legalized_hard_cost": 1.0,
                    "legalized_v_rel": 0.1,
                    "legalized_feasible": True,
                },
            },
            1600: {
                "observable": {"refiner_steps": 1600},
                "legalized": {
                    "blocks": 120,
                    "legalized_hard_cost": 2.0,
                    "legalized_v_rel": 0.2,
                    "legalized_feasible": True,
                },
            },
        }
    }

    result = module.fixed(cases, 1600)
    assert result["cases"] == 1
    assert abs(result["legalized_hard_cost"] - 2.0) < 1.0e-12

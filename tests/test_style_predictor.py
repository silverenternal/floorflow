import json

import numpy as np

from floorflow.inference.style_predictor import ContourStylePredictor


def test_contour_style_predictor_loads_and_predicts(tmp_path) -> None:
    path = tmp_path / "predictor.json"
    payload = {
        "input_mode": "contour_only",
        "center": [0.0] * 258,
        "scale": [1.0] * 258,
        "coefficients": [[0.0] * 258, [0.0] * 258],
        "intercepts": [1.0, 0.0],
        "classes": [0, 1],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    predictor = ContourStylePredictor.load(path)
    footprints = np.zeros((2, 16, 16), dtype=bool)
    footprints[:, 4:12, 4:12] = True
    assert predictor.predict(footprints).tolist() == [0, 0]

import numpy as np

from floorflow.evaluation.overlay_report import summarize_metric_rows


def test_overlay_rows_are_averaged_per_metric() -> None:
    summary = summarize_metric_rows(
        {
            "raw": [{"room_macro_iou": 0.1}, {"room_macro_iou": 0.3}],
            "overlay": [{"room_macro_iou": 0.5}, {"room_macro_iou": 0.7}],
        }
    )

    assert np.isclose(summary["raw"]["room_macro_iou"], 0.2)
    assert np.isclose(summary["overlay"]["room_macro_iou"], 0.6)

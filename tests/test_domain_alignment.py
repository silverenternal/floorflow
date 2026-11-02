import numpy as np
import pytest

from floorflow.data.domain_alignment import (
    class_alignment_rows,
    distribution_summary,
    normalized_pixel_distribution,
)


def test_distribution_metrics_and_missing_class_alignment() -> None:
    real = np.asarray([5.0, 5.0, 0.0])
    synthetic = np.asarray([8.0, 2.0, 0.0])
    assert distribution_summary(
        real / real.sum(),
        synthetic / synthetic.sum(),
    )["total_variation"] == pytest.approx(0.3)
    normalized = normalized_pixel_distribution(real, [0, 1])
    assert np.allclose(normalized, [0.5, 0.5])
    rows = class_alignment_rows(
        ("background", "room:Bedroom", "room:Bath"),
        [0, 1, 2],
        np.asarray([5.0, 4.0, 1.0]),
        np.asarray([8.0, 2.0, 0.0]),
    )
    assert rows[0]["class"] == "room:Bath"
    assert rows[0]["missing_in_synthetic"]

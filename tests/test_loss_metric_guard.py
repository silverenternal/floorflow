import pytest

from floorflow.training.loss_metric_guard import LossMetricGuard


def test_loss_metric_guard_fails_only_after_repeated_divergence() -> None:
    guard = LossMetricGuard(patience=2)
    guard.update(2.0, 0.20)
    first = guard.update(1.9, 0.18)
    assert first["diverged"]
    with pytest.raises(RuntimeError, match="declined for 2 epochs"):
        guard.update(1.8, 0.16)


def test_loss_metric_guard_resets_after_recovery() -> None:
    guard = LossMetricGuard(patience=2)
    guard.update(2.0, 0.20)
    guard.update(1.9, 0.18)
    recovered = guard.update(1.8, 0.19)
    assert recovered["divergence_count"] == 0

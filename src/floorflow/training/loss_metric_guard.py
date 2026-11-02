"""Fail-fast guard against repeated loss/metric divergence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossMetricGuard:
    metric_name: str = "room_macro_iou"
    loss_drop: float = 1.0e-3
    metric_drop: float = 1.0e-3
    patience: int = 2

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        self.previous_loss: float | None = None
        self.previous_metric: float | None = None
        self.divergence_count = 0

    def update(self, loss: float, metric: float) -> dict[str, object]:
        diverged = False
        if self.previous_loss is not None and self.previous_metric is not None:
            diverged = (
                loss < self.previous_loss - self.loss_drop
                and metric < self.previous_metric - self.metric_drop
            )
            self.divergence_count = (
                self.divergence_count + 1 if diverged else 0
            )
        self.previous_loss = float(loss)
        self.previous_metric = float(metric)
        if self.divergence_count >= self.patience:
            raise RuntimeError(
                f"{self.metric_name} declined for {self.divergence_count} "
                "epochs while training loss improved"
            )
        return {
            "metric_name": self.metric_name,
            "loss": self.previous_loss,
            "metric": self.previous_metric,
            "diverged": diverged,
            "divergence_count": self.divergence_count,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "previous_loss": self.previous_loss,
            "previous_metric": self.previous_metric,
            "divergence_count": self.divergence_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("metric_name", self.metric_name) != self.metric_name:
            raise ValueError("loss/metric guard configuration differs")
        self.previous_loss = state.get("previous_loss")
        self.previous_metric = state.get("previous_metric")
        self.divergence_count = int(state.get("divergence_count", 0))


__all__ = ["LossMetricGuard"]

"""Tests for predicted-control source parity auditing."""

from __future__ import annotations

import pytest
import torch

from floorflow.evaluation.control_source_parity import (
    _binary_counts,
    _manifest_audit,
    _target_label_path,
    control_fidelity_row,
    relative_equivalence_verdict,
)


class _Taxonomy:
    room_ids = (1, 2)
    opening_ids = (5,)
    furniture_ids = (6,)

    @staticmethod
    def class_id(name: str) -> int:
        assert name == "wall"
        return 3


def test_binary_counts_reports_precision_recall_and_iou() -> None:
    result = _binary_counts(
        torch.tensor([True, True, False, False]),
        torch.tensor([True, False, True, False]),
    )
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["iou"] == 1.0 / 3.0


def test_manifest_audit_ignores_runtime_metadata() -> None:
    reference = {
        "schema_version": "v1",
        "source": "predicted",
        "analysis_variant": "contour_cfm_guided",
        "target_used_for_control": False,
        "deployable": True,
        "flow_steps": 12,
        "noise_scale": 0.75,
        "control_source": "blend",
        "checkpoint_sha256": "abc",
        "taxonomy_sha256": "def",
        "control_sizes": {"room": 224},
        "created_at": "old",
    }
    candidate = dict(reference, created_at="new")
    assert _manifest_audit(reference, candidate)["passed"]


def test_control_fidelity_row_has_contract_metrics() -> None:
    labels = torch.tensor(
        [
            [1, 1, 3, 3],
            [1, 1, 5, 3],
            [2, 2, 6, 6],
            [2, 2, 6, 6],
        ]
    )
    controls = {
        "room_logits": torch.zeros(2, 4, 4),
        "program": torch.tensor([True, True]),
        "structure": torch.zeros(5, 4, 4),
        "furniture": torch.zeros(4, 4, 4),
        "support": torch.zeros(5, 4, 4),
    }
    controls["room_logits"][0, :2] = 1.0
    controls["room_logits"][1, 2:] = 1.0
    controls["structure"][2] = 1.0
    controls["structure"][3] = 1.0
    controls["furniture"][2] = 1.0
    row = control_fidelity_row(controls, labels, _Taxonomy())
    assert 0.0 <= row["room_macro_iou"] <= 1.0
    assert 0.0 <= row["wall_f1"] <= 1.0
    assert 0.0 <= row["opening_f1"] <= 1.0
    assert "program_exact" in row


def test_target_label_path_prefers_nested_full_cache(tmp_path) -> None:
    nested = tmp_path / "train" / "category" / "12" / "Floor_1.npz"
    nested.parent.mkdir(parents=True)
    nested.touch()
    assert (
        _target_label_path(tmp_path, "train", "category/12__Floor-1")
        == nested
    )


def test_relative_equivalence_uses_ten_percent_with_one_point_floor() -> None:
    low = relative_equivalence_verdict(
        {
            "baseline_mean": 0.04,
            "mean_delta": -0.009,
            "ci_low": -0.0095,
            "ci_high": -0.008,
        }
    )
    high = relative_equivalence_verdict(
        {
            "baseline_mean": 0.40,
            "mean_delta": -0.025,
            "ci_low": -0.030,
            "ci_high": -0.020,
        }
    )
    assert low == {"margin": 0.01, "passed": True}
    assert high["margin"] == pytest.approx(0.04)
    assert high["passed"]

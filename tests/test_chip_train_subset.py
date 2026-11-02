"""Tests for deterministic chip training subsets."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

from floorflow.chip_flow.model import ChipBlockFlowModel
from floorflow.chip_flow.train_cli import validate_training_scope
from floorflow.chip_flow.train_evaluation import (
    BalancedBatchSampler,
    evaluate,
    subset_indices_by_count,
    validate_disjoint_splits,
)
from floorflow.chip_flow.training_manifest import (
    RESUME_CRITICAL_KEYS,
    RESUME_NEW_PARAM_PREFIXES,
    validate_resume_config,
)
from floorflow.chip_flow.training_scopes import (
    ChannelStageController,
    parse_scope_schedule,
)
from floorflow.chip_flow.training_validation import (
    validate_atom_metric_objective_schedule,
)


class FakeStore:
    by_block_count: ClassVar[dict[int, np.ndarray]] = {
        21: np.asarray([0, 2, 4], dtype=np.int64),
        22: np.asarray([1, 3, 5], dtype=np.int64),
    }

    def __len__(self) -> int:
        return 6

    def indices(self, block_count: int) -> np.ndarray:
        return self.by_block_count[block_count]


def test_training_subset_is_grouped_by_block_count() -> None:
    grouped = subset_indices_by_count(FakeStore(), {1, 2, 4})

    np.testing.assert_array_equal(grouped[21], [2, 4])
    np.testing.assert_array_equal(grouped[22], [1])


def test_training_subset_rejects_empty_and_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        subset_indices_by_count(FakeStore(), set())
    with pytest.raises(ValueError, match="out-of-range"):
        subset_indices_by_count(FakeStore(), {6})


def test_boundary_anchor_is_resume_allowlisted() -> None:
    assert "boundary_anchor." in RESUME_NEW_PARAM_PREFIXES
    assert "stage_lr_restart" in RESUME_CRITICAL_KEYS
    assert "stage_lr_warmup_steps" in RESUME_CRITICAL_KEYS


def _split_store(path: str, indices: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={"source_paths": [path]},
        arrays={
            "source_id": np.zeros(len(indices), dtype=np.int64),
            "source_index": np.asarray(indices, dtype=np.int64),
        },
    )


def test_train_dev_split_overlap_is_rejected() -> None:
    train = _split_store("shard.th", [0, 1])
    validate_disjoint_splits(train, _split_store("shard.th", [2, 3]))

    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_splits(train, _split_store("shard.th", [1, 2]))


def test_training_scope_requires_explicit_full_data_override() -> None:
    base = {"batch_size": 128, "grad_accum_steps": 1}
    with pytest.raises(ValueError, match="allow-full-data"):
        validate_training_scope(
            SimpleNamespace(
                train_indices=None,
                allow_full_data=False,
                **base,
            )
        )
    validate_training_scope(
        SimpleNamespace(
            train_indices=object(),
            allow_full_data=False,
            **base,
        )
    )
    validate_training_scope(
        SimpleNamespace(
            train_indices=None,
            allow_full_data=True,
            **base,
        )
    )
    with pytest.raises(ValueError, match="gradient accumulation"):
        validate_training_scope(
            SimpleNamespace(
                train_indices=object(),
                allow_full_data=False,
                batch_size=128,
                grad_accum_steps=0,
            )
        )


def test_training_scope_fails_closed_on_inactive_v2_closures() -> None:
    base = {
        "train_indices": object(),
        "allow_full_data": False,
        "batch_size": 128,
        "grad_accum_steps": 1,
    }
    with pytest.raises(ValueError, match="positive legalizer response weight"):
        validate_training_scope(
            SimpleNamespace(
                structured_legalizer_apply=True,
                legalizer_response_weight=0.0,
                response_sidecar=object(),
                **base,
            )
        )
    with pytest.raises(ValueError, match="response loss requires"):
        validate_training_scope(
            SimpleNamespace(
                legalizer_response_weight=0.1,
                response_sidecar=None,
                **base,
            )
        )
    with pytest.raises(ValueError, match="already own hierarchy coupling"):
        validate_training_scope(
            SimpleNamespace(
                structured_heads=True,
                membership_hierarchy_conditioning=True,
                **base,
            )
        )
    with pytest.raises(ValueError, match="already own legalizer response"):
        validate_training_scope(
            SimpleNamespace(
                structured_heads=True,
                standalone_legalizer_response=True,
                **base,
            )
        )
    with pytest.raises(ValueError, match="atom isolation"):
        validate_training_scope(
            SimpleNamespace(
                require_atom_isolation=True,
                training_scope_schedule=None,
                **base,
            )
        )
    with pytest.raises(ValueError, match="same-call structured hierarchy"):
        validate_training_scope(
            SimpleNamespace(
                coarse_commit_fraction=0.3,
                structured_heads=False,
                **base,
            )
        )
    with pytest.raises(ValueError, match="typed_mode=dense"):
        validate_training_scope(
            SimpleNamespace(
                relation_specific_attention=True,
                typed_mode="dense",
                **base,
            )
        )
    with pytest.raises(ValueError, match="legalizer-target"):
        validate_training_scope(
            SimpleNamespace(
                endpoint_legalizer_target_weight=0.1,
                response_sidecar=None,
                **base,
            )
        )


def test_atom_objectives_must_activate_inside_their_owned_stage() -> None:
    schedule = parse_scope_schedule(
        "0:area,150:b2b,300:port_route,450:boundary,600:tail,750:fusion"
    )
    with pytest.raises(ValueError, match="routing_proxy delay 1000"):
        validate_atom_metric_objective_schedule(
            SimpleNamespace(
                official_cost_final_weight=0.1,
                official_cost_delay_steps=25,
                grouping_proxy_final_weight=0.01,
                grouping_proxy_delay_steps=150,
                routing_proxy_final_weight=0.01,
                routing_proxy_delay_steps=1000,
            ),
            schedule,
        )
    with pytest.raises(ValueError, match="starts before the port_route stage"):
        validate_atom_metric_objective_schedule(
            SimpleNamespace(
                official_cost_final_weight=0.1,
                official_cost_delay_steps=25,
                grouping_proxy_final_weight=0.01,
                grouping_proxy_delay_steps=150,
                routing_proxy_final_weight=0.01,
                routing_proxy_delay_steps=250,
            ),
            schedule,
        )
    validate_atom_metric_objective_schedule(
        SimpleNamespace(
            official_cost_final_weight=0.1,
            official_cost_delay_steps=25,
            grouping_proxy_final_weight=0.01,
            grouping_proxy_delay_steps=150,
            routing_proxy_final_weight=0.01,
            routing_proxy_delay_steps=300,
        ),
        schedule,
    )


def test_atom_isolated_lr_restarts_at_each_stage() -> None:
    schedule_text = (
        "0:area,150:b2b,300:port_route,450:boundary,600:tail,750:fusion"
    )
    args = SimpleNamespace(
        training_scope="all",
        training_scope_schedule=schedule_text,
        stage_lr_restart=True,
        stage_lr_warmup_steps=0,
        lr=1.0e-4,
        steps=1000,
        warmup=50,
    )
    controller = ChannelStageController.from_args(args)
    tail_end = controller.learning_rate(749, args)
    fusion_start = controller.learning_rate(750, args)

    assert fusion_start == pytest.approx(1.0e-4)
    assert tail_end < fusion_start


def test_weights_only_resume_allows_changed_training_schedule() -> None:
    saved = {"hierarchy_source_mode": "random", "ema_mode": "fixed"}
    current = {"hierarchy_source_mode": "mixed", "ema_mode": "fixed"}

    validate_resume_config(saved, current, weights_only=True)
    with pytest.raises(ValueError, match="hierarchy_source_mode"):
        validate_resume_config(saved, current, weights_only=False)


@pytest.mark.parametrize(
    "key",
    (
        "routing_proxy_final_weight",
        "grouping_proxy_final_weight",
        "legalizer_response_weight",
        "endpoint_compatibility_weight",
    ),
)
def test_resume_rejects_metric_objective_schedule_drift(key: str) -> None:
    saved = {key: 0.01}
    current = {key: 0.02}

    with pytest.raises(ValueError, match=key):
        validate_resume_config(saved, current, weights_only=False)


def test_balanced_batch_sampler_is_deterministic_and_count_balanced() -> None:
    groups = {
        21: np.asarray([0, 2, 4], dtype=np.int64),
        22: np.asarray([1, 3, 5], dtype=np.int64),
    }
    first = BalancedBatchSampler(groups, batch_size=2, seed=23)
    second = BalancedBatchSampler(groups, batch_size=2, seed=23)

    first_batches = [first.next_indices() for _ in range(6)]
    second_batches = [second.next_indices() for _ in range(6)]

    assert [count for count, _ in first_batches[:4]] == [21, 22, 21, 22]
    for (count_a, indices_a), (count_b, indices_b) in zip(
        first_batches,
        second_batches,
        strict=True,
    ):
        assert count_a == count_b
        np.testing.assert_array_equal(indices_a, indices_b)
        assert set(indices_a.tolist()).issubset(set(groups[count_a].tolist()))


def test_balanced_batch_sampler_state_restores_exact_resume_order() -> None:
    groups = {
        21: np.asarray([0, 2, 4, 6], dtype=np.int64),
        22: np.asarray([1, 3, 5, 7], dtype=np.int64),
    }
    sampler = BalancedBatchSampler(groups, batch_size=3, seed=23)
    [sampler.next_indices() for _ in range(3)]
    state = sampler.state_dict()
    expected = [sampler.next_indices() for _ in range(4)]

    restored = BalancedBatchSampler(groups, batch_size=3, seed=999)
    restored.load_state_dict(state)
    actual = [restored.next_indices() for _ in range(4)]

    for (expected_count, expected_indices), (actual_count, actual_indices) in zip(
        expected,
        actual,
        strict=True,
    ):
        assert actual_count == expected_count
        np.testing.assert_array_equal(actual_indices, expected_indices)


class _SingleCaseStore:
    by_block_count: ClassVar = {1: np.asarray([0], dtype=np.int64)}
    arrays: ClassVar = {"n": np.asarray([1], dtype=np.int64)}

    def indices(self, block_count=None):
        return self.by_block_count[1]

    def batch(self, indices, *, device):
        return {
            "n": 1,
            "areas": torch.ones(1, 1, device=device),
            "source_latent": torch.zeros(1, 1, 3, device=device),
            "target_latent": torch.zeros(1, 1, 3, device=device),
            "constraints": torch.zeros(1, 1, 5, device=device),
            "features": torch.zeros(1, 1, 10, device=device),
            "b2b": torch.zeros(1, 1, 3, device=device),
            "p2b": torch.zeros(1, 1, 3, device=device),
            "pins": torch.zeros(1, 1, 2, device=device),
            "n_b2b": torch.zeros(1, dtype=torch.long, device=device),
            "n_p2b": torch.zeros(1, dtype=torch.long, device=device),
            "n_pins": torch.zeros(1, dtype=torch.long, device=device),
        }


@pytest.mark.parametrize("training", [True, False])
def test_evaluate_restores_model_training_mode(training: bool) -> None:
    model = ChipBlockFlowModel(
        feature_dim=10,
        hidden=16,
        layers=1,
        heads=4,
        top_k=None,
    )
    model.train(training)

    evaluate(
        _SingleCaseStore(),
        np.asarray([0], dtype=np.int64),
        model,
        device=torch.device("cpu"),
        batch_size=1,
        sampling_steps=1,
        seed=19,
    )

    assert model.training is training

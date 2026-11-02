"""Attach canonical verifier rewards to training batches."""

from __future__ import annotations

import torch

from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.objectives.token_reward import token_candidate_reward
from floorflow.training.rplan_generator_contract import contract_from_output


def _slice_batch(batch: dict, row: int, batch_size: int) -> dict:
    return {
        key: value[row : row + 1]
        if torch.is_tensor(value)
        and value.ndim > 0
        and value.shape[0] == batch_size
        else value
        for key, value in batch.items()
    }


def _slice_output(output: dict, row: int, batch_size: int) -> dict:
    return {
        key: value[row : row + 1]
        if torch.is_tensor(value)
        and value.ndim > 0
        and value.shape[0] == batch_size
        else value
        for key, value in output.items()
    }


def attach_verifier_rewards(
    output: dict,
    batch: dict,
    taxonomy,
    tolerances: FEVPTolerances,
    *,
    max_samples: int | None = None,
) -> torch.Tensor:
    batch_size = int(batch["boundary_mask"].shape[0])
    count = batch_size if max_samples is None else min(batch_size, max_samples)
    rewards = output["room_exists"].new_zeros(batch_size)
    for row in range(count):
        row_batch = _slice_batch(batch, row, batch_size)
        row_output = _slice_output(output, row, batch_size)
        contract = contract_from_output(
            row_output,
            row_batch,
            taxonomy,
            tolerances,
        )
        expected_rooms = int(batch["room_exists"][row].sum())
        reward, _components = token_candidate_reward(
            contract,
            {},
            expected_room_count=expected_rooms,
        )
        rewards[row] = float(reward)
    if count and count < batch_size:
        rewards[count:] = rewards[:count].mean()
    return rewards


__all__ = ["attach_verifier_rewards"]

"""Unified policy contract for dynamic masks, budgets and remasking."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.flow.mask_policies import AdversarialMaskScheduler, MaskPolicy
from floorflow.flow.masked_control import control_channel_groups
from floorflow.model.learned_mask_generator import LearnedMaskGenerator
from floorflow.training.masked_conditioning import (
    PreparedCondition,
    prepare_masked_condition,
)


@dataclass(frozen=True)
class DynamicPolicyOutput:
    condition: PreparedCondition
    family_budget: dict[str, torch.Tensor]
    policy_name: str | None


class RoomDynamicPolicy:
    """Thin adapter around existing mask, scheduler and budget implementations."""

    def __init__(
        self,
        taxonomy: FloorplanTaxonomy,
        *,
        mask_generator: LearnedMaskGenerator | None = None,
        scheduler: AdversarialMaskScheduler | None = None,
        fixed_policy: MaskPolicy | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.mask_generator = mask_generator
        self.scheduler = scheduler
        self.fixed_policy = fixed_policy
        self._best_score = float("-inf")
        self._best_state: dict[str, object] | None = None

    def propose(
        self,
        source: torch.Tensor,
        footprint: torch.Tensor,
        missing_token: torch.Tensor,
        *,
        missing_ratio: float,
        floor_ids: torch.Tensor | None = None,
    ) -> DynamicPolicyOutput:
        prepared = prepare_masked_condition(
            source,
            footprint,
            self.taxonomy,
            missing_token,
            missing_ratio=missing_ratio,
            floor_ids=floor_ids,
            mask_generator=self.mask_generator,
            mask_policy=self.fixed_policy,
            scheduler=self.scheduler,
        )
        groups = control_channel_groups(self.taxonomy)
        family_budget = {}
        for name, channels in groups.items():
            channels = channels.to(prepared.missing_mask.device)
            family_budget[name] = (
                prepared.missing_mask[:, channels]
                .sum(dim=(1, 2, 3))
                .float()
            )
        return DynamicPolicyOutput(
            condition=prepared,
            family_budget=family_budget,
            policy_name=prepared.policy_name,
        )

    def update(self, hardness: float, *, policy_name: str | None = None) -> None:
        if self.scheduler is not None and policy_name is not None:
            self.scheduler.update(policy_name, hardness)

    def observe_score(self, score: float) -> None:
        if float(score) > self._best_score:
            self._best_score = float(score)
            self._best_state = self.state_dict()

    def rollback(self) -> None:
        if self._best_state is not None:
            self.load_state_dict(self._best_state)

    def state_dict(self) -> dict[str, object]:
        return {
            "mask_generator": (
                deepcopy(self.mask_generator.state_dict())
                if self.mask_generator is not None
                else None
            ),
            "scheduler": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "best_score": self._best_score,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        generator = state.get("mask_generator")
        if self.mask_generator is not None and isinstance(generator, dict):
            self.mask_generator.load_state_dict(generator)
        scheduler = state.get("scheduler")
        if self.scheduler is not None and isinstance(scheduler, dict):
            self.scheduler.load_state_dict(scheduler)
        self._best_score = float(state.get("best_score", float("-inf")))


__all__ = ["DynamicPolicyOutput", "RoomDynamicPolicy"]

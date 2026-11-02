"""Composable dynamic-mask policies for sparse coarse CAD controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.flow.masked_control import (
    ControlMask,
    control_channel_groups,
    sample_spatial_mask,
)


class MaskPolicy(Protocol):
    name: str

    def sample(
        self,
        source: torch.Tensor,
        footprint: torch.Tensor,
        taxonomy: FloorplanTaxonomy,
        *,
        ratio: float | None,
        generator: torch.Generator | None = None,
    ) -> ControlMask: ...


def _ratios(
    source: torch.Tensor,
    ratio: float | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if ratio is None:
        return torch.empty(source.shape[0], device=source.device).uniform_(
            0.15,
            0.9,
            generator=generator,
        )
    return torch.full(
        (source.shape[0],),
        float(ratio),
        device=source.device,
        dtype=source.dtype,
    )


@dataclass(frozen=True)
class SpatialBlockMaskPolicy:
    name: str = "spatial"

    def sample(
        self,
        source: torch.Tensor,
        footprint: torch.Tensor,
        taxonomy: FloorplanTaxonomy,
        *,
        ratio: float | None,
        generator: torch.Generator | None = None,
    ) -> ControlMask:
        del taxonomy
        ratios = _ratios(source, ratio, generator)
        spatial = sample_spatial_mask(
            source,
            footprint,
            ratios,
            generator=generator,
        )
        return ControlMask(
            value=spatial,
            spatial=spatial,
            group=torch.zeros_like(spatial, dtype=torch.bool),
            ratio=ratios,
        )


@dataclass(frozen=True)
class AttributeMaskPolicy:
    """MaskPLAN-style independent partial inputs for each attribute group."""

    name: str = "attribute"
    min_room_channels: int = 1

    def sample(
        self,
        source: torch.Tensor,
        footprint: torch.Tensor,
        taxonomy: FloorplanTaxonomy,
        *,
        ratio: float | None,
        generator: torch.Generator | None = None,
    ) -> ControlMask:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        ratios = _ratios(source, ratio, generator)
        groups = control_channel_groups(taxonomy)
        channel_mask = torch.zeros(
            source.shape[0],
            source.shape[1],
            device=source.device,
            dtype=torch.bool,
        )
        for index in range(source.shape[0]):
            for values in groups.values():
                values = values.to(source.device)
                count = min(
                    len(values),
                    max(
                        0,
                        int(round(len(values) * float(ratios[index]))),
                    ),
                )
                if count:
                    selected = torch.randperm(
                        len(values),
                        device=source.device,
                        generator=generator,
                    )[:count]
                    channel_mask[index, values[selected]] = True
            room_channels = groups["room"].to(source.device)
            if int((~channel_mask[index, room_channels]).sum()) < self.min_room_channels:
                keep = room_channels[
                    torch.randint(
                        0,
                        len(room_channels),
                        (1,),
                        device=source.device,
                        generator=generator,
                    )
                ]
                channel_mask[index, keep] = False
        spatial = torch.zeros(
            source.shape[0],
            1,
            *source.shape[-2:],
            device=source.device,
            dtype=torch.bool,
        )
        group = channel_mask[:, :, None, None].expand_as(source) & footprint.bool()
        return ControlMask(
            value=group,
            spatial=spatial,
            group=group,
            ratio=ratios,
        )


@dataclass(frozen=True)
class SalienceMaskPolicy:
    """Mask low-salience channel-spatial controls first."""

    name: str = "salience"
    min_room_channels: int = 1

    def sample(
        self,
        source: torch.Tensor,
        footprint: torch.Tensor,
        taxonomy: FloorplanTaxonomy,
        *,
        ratio: float | None,
        generator: torch.Generator | None = None,
    ) -> ControlMask:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        ratios = _ratios(source, ratio, generator)
        salience = source.float().abs()
        salience = salience + salience.mean(dim=(2, 3), keepdim=True)
        region = footprint.expand_as(source).bool()
        hard = torch.zeros_like(source, dtype=torch.bool)
        for index in range(source.shape[0]):
            valid = region[index]
            values = salience[index][valid]
            if values.numel() == 0:
                continue
            count = min(
                values.numel(),
                max(1, int(round(values.numel() * float(ratios[index])))),
            )
            threshold = values.topk(count, largest=False).values[-1]
            hard[index] = (salience[index] <= threshold) & valid
            room_channels = torch.tensor(
                taxonomy.room_ids,
                device=source.device,
                dtype=torch.long,
            ) - 1
            if int((~hard[index, room_channels]).sum()) < self.min_room_channels:
                hard[index, room_channels[0]] = False
        spatial = hard.amax(dim=1, keepdim=True)
        return ControlMask(
            value=hard,
            spatial=spatial,
            group=hard,
            ratio=ratios,
        )


@dataclass(frozen=True)
class CompositeMaskPolicy:
    name: str = "composite"
    spatial_probability: float = 0.5

    def sample(
        self,
        source: torch.Tensor,
        footprint: torch.Tensor,
        taxonomy: FloorplanTaxonomy,
        *,
        ratio: float | None,
        generator: torch.Generator | None = None,
    ) -> ControlMask:
        spatial = SpatialBlockMaskPolicy().sample(
            source,
            footprint,
            taxonomy,
            ratio=ratio,
            generator=generator,
        )
        attribute = AttributeMaskPolicy().sample(
            source,
            footprint,
            taxonomy,
            ratio=ratio,
            generator=generator,
        )
        use_spatial = torch.rand(1, device=source.device, generator=generator) < float(
            self.spatial_probability
        )
        if bool(use_spatial):
            value = spatial.value | attribute.value
        else:
            value = attribute.value
        return ControlMask(
            value=value,
            spatial=spatial.spatial,
            group=attribute.group,
            ratio=attribute.ratio,
        )


@dataclass
class AdversarialMaskScheduler:
    """Exp3-style online selection over mask policies by reconstruction hardness."""

    policies: tuple[MaskPolicy, ...]
    learning_rate: float = 0.25
    reward_ema: float = 0.90
    minimum_probability: float = 0.03
    weights: dict[str, float] = field(init=False)
    reward_means: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.weights = {policy.name: 1.0 for policy in self.policies}
        self.reward_means = {policy.name: 0.5 for policy in self.policies}
        if not 0.0 <= self.reward_ema < 1.0:
            raise ValueError("reward_ema must be in [0,1)")
        if not 0.0 <= self.minimum_probability < 1.0 / len(self.policies):
            raise ValueError("minimum_probability leaves no probability mass")

    def sample(self) -> MaskPolicy:
        names = tuple(self.weights)
        weights = torch.tensor([self.weights[name] for name in names])
        index = int(torch.multinomial(weights / weights.sum(), 1))
        return self.policies[index]

    def update(self, name: str, difficulty: float) -> None:
        clipped = min(1.0, max(0.0, float(difficulty)))
        self.reward_means[name] = (
            self.reward_ema * self.reward_means[name]
            + (1.0 - self.reward_ema) * clipped
        )
        self.weights[name] *= float(
            torch.exp(
                torch.tensor(
                    self.learning_rate * (self.reward_means[name] - 0.5)
                )
            ).item()
        )
        total = sum(self.weights.values())
        self.weights = {
            key: max(1.0e-6, value / total)
            for key, value in self.weights.items()
        }
        count = len(self.weights)
        floor = min(float(self.minimum_probability), 1.0 / count)
        free_mass = 1.0 - floor * count
        free_total = sum(
            max(0.0, value - floor) for value in self.weights.values()
        )
        if free_total > 0:
            self.weights = {
                key: floor + free_mass * max(0.0, value - floor) / free_total
                for key, value in self.weights.items()
            }

    def state_dict(self) -> dict[str, object]:
        return {
            "type": "adversarial",
            "weights": dict(self.weights),
            "reward_means": dict(self.reward_means),
            "learning_rate": self.learning_rate,
            "reward_ema": self.reward_ema,
            "minimum_probability": self.minimum_probability,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        weights = state.get("weights")
        if not isinstance(weights, dict):
            raise ValueError("scheduler state has no policy weights")
        if set(weights) != set(self.weights):
            raise ValueError("scheduler policies do not match checkpoint")
        values = torch.tensor(
            [float(weights[name]) for name in self.weights],
            dtype=torch.float32,
        ).clamp_min(1.0e-8)
        values = values / values.sum()
        self.weights = {
            name: float(values[index])
            for index, name in enumerate(self.weights)
        }
        reward_means = state.get("reward_means", self.reward_means)
        if not isinstance(reward_means, dict):
            raise ValueError("scheduler reward means must be a mapping")
        self.reward_means = {
            name: float(reward_means.get(name, 0.5))
            for name in self.reward_means
        }


def make_mask_policy(name: str) -> MaskPolicy:
    if name == "spatial":
        return SpatialBlockMaskPolicy()
    if name == "attribute":
        return AttributeMaskPolicy()
    if name == "salience":
        return SalienceMaskPolicy()
    if name == "composite":
        return CompositeMaskPolicy()
    raise ValueError(f"unknown mask policy: {name}")


__all__ = [
    "AdversarialMaskScheduler",
    "AttributeMaskPolicy",
    "CompositeMaskPolicy",
    "MaskPolicy",
    "SalienceMaskPolicy",
    "SpatialBlockMaskPolicy",
    "make_mask_policy",
]

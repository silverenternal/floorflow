"""Dev-curve selection and chip-flow evaluation shared by training runs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from floorflow.chip_flow.batched import batch_physics_terms
from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.flow_adapter import (
    sample_for_batch,
    sparse_diagnostic_totals,
)
from floorflow.chip_flow.model import decode_boxes_batch, encode_latents
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.proposals import TokenOTCache, token_ot_reorder_batch
from floorflow.chip_flow.slot_model import ChipSlotFlowModel
from floorflow.chip_physics import group_penalty_batch


def select_curve_indices(
    store: PackedCaseStore,
    *,
    per_count: int,
    seed: int,
    exclude_indices: set[int] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    excluded = np.asarray(sorted(exclude_indices or ()), dtype=np.int64)
    selected = []
    for count in sorted(store.by_block_count):
        indices = store.indices(count)
        if excluded.size:
            indices = np.setdiff1d(indices, excluded, assume_unique=True)
        if len(indices) < per_count:
            raise ValueError(
                f"block-count stratum {count} has fewer than {per_count} cases"
            )
        selected.extend(
            rng.choice(indices, per_count, replace=False)
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def subset_indices_by_count(
    store: PackedCaseStore,
    indices: set[int],
) -> dict[int, np.ndarray]:
    """Return non-empty allowed training indices grouped by block count."""

    selected = np.asarray(sorted(indices), dtype=np.int64)
    if selected.size == 0:
        raise ValueError("training subset must not be empty")
    if selected.min() < 0 or selected.max() >= len(store):
        raise ValueError("training subset contains an out-of-range index")
    by_count = {}
    for count in sorted(store.by_block_count):
        allowed = np.intersect1d(
            store.indices(count),
            selected,
            assume_unique=True,
        )
        if allowed.size:
            by_count[int(count)] = allowed
    if not by_count:
        raise ValueError("training subset contains no cached cases")
    return by_count


def validate_disjoint_splits(
    train: PackedCaseStore,
    dev: PackedCaseStore,
) -> None:
    """Reject train/dev overlap using stable source and layout identifiers."""

    def keys(store: PackedCaseStore) -> set[tuple[str, int]]:
        paths = store.metadata.get("source_paths")
        if not isinstance(paths, list):
            raise TypeError("packed split metadata lacks source_paths")
        return {
            (str(paths[int(source)]), int(layout))
            for source, layout in zip(
                store.arrays["source_id"],
                store.arrays["source_index"],
                strict=True,
            )
        }

    overlap = keys(train) & keys(dev)
    if overlap:
        raise ValueError(f"train/dev split overlap contains {len(overlap)} cases")


class BalancedBatchSampler:
    """Cycle block counts and draw cases without replacement within a queue."""

    def __init__(
        self,
        allowed_by_count: Mapping[int, np.ndarray],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not allowed_by_count:
            raise ValueError("balanced sampler needs at least one block count")
        self.groups = {
            int(count): np.asarray(indices, dtype=np.int64)
            for count, indices in allowed_by_count.items()
        }
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.counts = tuple(sorted(self.groups))
        self.queues = {count: [] for count in self.counts}
        self.cursor = 0

    def state_dict(self) -> dict:
        """Return the exact sampler state needed for deterministic resume."""

        return {
            "counts": self.counts,
            "cursor": self.cursor,
            "rng_state": self.rng.bit_generator.state,
            "queues": {
                count: list(values)
                for count, values in self.queues.items()
            },
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore a sampler state after validating its block-count layout."""

        counts = tuple(int(value) for value in state["counts"])
        if counts != self.counts:
            raise ValueError("balanced sampler block-count layout mismatch")
        cursor = int(state["cursor"])
        if cursor < 0:
            raise ValueError("balanced sampler cursor must be non-negative")
        queues = {
            int(count): [int(value) for value in values]
            for count, values in dict(state["queues"]).items()
        }
        if set(queues) != set(self.counts):
            raise ValueError("balanced sampler queue layout mismatch")
        self.cursor = cursor
        self.rng.bit_generator.state = state["rng_state"]
        self.queues = queues

    def next_indices(self) -> tuple[int, np.ndarray]:
        count = self.counts[self.cursor % len(self.counts)]
        self.cursor += 1
        queue = self.queues[count]
        while len(queue) < self.batch_size:
            queue.extend(int(value) for value in self.rng.permutation(self.groups[count]))
        indices = np.asarray(queue[: self.batch_size], dtype=np.int64)
        del queue[: self.batch_size]
        return count, indices


def sample_batch(
    store: PackedCaseStore,
    rng: np.random.Generator,
    *,
    batch_size: int,
    device: torch.device,
    geometry_augment: str = "none",
    geometry_augment_probability: float = 1.0,
    allowed_by_count: Mapping[int, np.ndarray] | None = None,
    preselected_indices: np.ndarray | None = None,
) -> dict:
    if preselected_indices is None:
        groups = allowed_by_count or store.by_block_count
        count = int(rng.choice(sorted(groups)))
        candidates = np.asarray(groups[count], dtype=np.int64)
        indices = rng.choice(
            candidates,
            size=batch_size,
            replace=len(candidates) < batch_size,
        )
    else:
        indices = np.asarray(preselected_indices, dtype=np.int64)
        if indices.size != batch_size:
            raise ValueError("preselected indices must match batch_size")
    batch = store.batch(indices, device=device)
    batch["_indices"] = np.asarray(indices, dtype=np.int64)
    return augment_geometry(
        batch,
        geometry_augment,
        rng,
        probability=geometry_augment_probability,
    )


def _reflect_axis(batch: dict, axis: int, mask: torch.Tensor) -> None:
    boxes = batch["boxes"]
    lower = boxes[..., axis].min(dim=1).values
    upper = boxes[..., axis] + boxes[..., 2 + axis]
    upper = upper.max(dim=1).values
    reflected = (
        lower[:, None]
        + upper[:, None]
        - (boxes[..., axis] + boxes[..., 2 + axis])
    )
    boxes[..., axis] = torch.where(mask[:, None], reflected, boxes[..., axis])

    pins = batch["pins"]
    pin_rows = torch.arange(pins.shape[1], device=pins.device)[None, :]
    pin_mask = pin_rows < batch["n_pins"][:, None]
    reflected_pins = lower[:, None] + upper[:, None] - pins[..., axis]
    pins[..., axis] = torch.where(
        mask[:, None] & pin_mask,
        reflected_pins,
        pins[..., axis],
    )

    codes = batch["constraints"][..., 4].to(dtype=torch.long)
    if axis == 0:
        swapped = ((codes & 1) << 1) | ((codes & 2) >> 1) | (codes & 12)
    else:
        swapped = ((codes & 4) << 1) | ((codes & 8) >> 1) | (codes & 3)
    batch["constraints"][..., 4] = torch.where(
        mask[:, None],
        swapped.to(dtype=batch["constraints"].dtype),
        batch["constraints"][..., 4],
    )


def _refresh_pin_features(batch: dict) -> None:
    features = batch["features"]
    if features.shape[-1] < 9:
        raise ValueError("geometry augmentation requires pin-mean feature columns")
    pins = batch["pins"]
    pin_rows = torch.arange(pins.shape[1], device=pins.device)[None, :]
    pin_mask = pin_rows < batch["n_pins"][:, None]
    counts = batch["n_pins"].clamp_min(1).to(dtype=pins.dtype)
    pin_mean = (pins * pin_mask[..., None]).sum(dim=1) / counts[:, None]
    scale = batch["areas"].sum(dim=1).sqrt().clamp_min(1.0e-6)
    pin_pull = (pin_mean / scale[:, None]).clamp(-2.0, 2.0)
    features[..., 7:9] = pin_pull[:, None, :]


def augment_geometry(
    batch: dict,
    mode: str,
    rng: np.random.Generator,
    probability: float = 1.0,
) -> dict:
    """Apply label-preserving chip reflections to an existing training batch."""

    if mode == "none":
        return batch
    if not 0.0 <= probability <= 1.0:
        raise ValueError("geometry augmentation probability must be in [0, 1]")
    choices = {
        "hflip": ("hflip",),
        "vflip": ("vflip",),
        "rot180": ("rot180",),
        "dihedral": ("none", "hflip", "vflip", "rot180"),
    }
    if mode not in choices:
        raise ValueError(f"unknown geometry augmentation: {mode}")
    batch = dict(batch)
    for name in ("boxes", "pins", "constraints", "features"):
        batch[name] = batch[name].clone()
    selected = np.asarray(choices[mode], dtype=object)[
        rng.integers(0, len(choices[mode]), size=batch["boxes"].shape[0])
    ]
    selected[rng.random(batch["boxes"].shape[0]) >= probability] = "none"
    flip_x = torch.as_tensor(
        np.isin(selected, ("hflip", "rot180")),
        dtype=torch.bool,
        device=batch["boxes"].device,
    )
    flip_y = torch.as_tensor(
        np.isin(selected, ("vflip", "rot180")),
        dtype=torch.bool,
        device=batch["boxes"].device,
    )
    _reflect_axis(batch, 0, flip_x)
    _reflect_axis(batch, 1, flip_y)
    batch["target_latent"] = encode_latents(batch["boxes"], batch["areas"])
    _refresh_pin_features(batch)
    batch["geometry_augmentation"] = selected
    return batch


@torch.no_grad()
def evaluate(
    store: PackedCaseStore,
    indices: np.ndarray,
    model: ChipSlotFlowModel,
    *,
    device: torch.device,
    batch_size: int,
    sampling_steps: int,
    seed: int,
    alignment_cache: TokenOTCache | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    diagnostic_totals: dict[str, torch.Tensor] = {}
    metric_totals: torch.Tensor | None = None
    metric_names: tuple[str, ...] | None = None
    cases = 0
    case_counts = store.arrays["n"][indices]
    for count in np.unique(case_counts):
        count_indices = indices[case_counts == count]
        for start in range(0, len(count_indices), batch_size):
            batch = store.batch(
                count_indices[start : start + batch_size],
                device=device,
            )
            latent, diagnostics = sample_for_batch(
                model,
                batch,
                sampling_steps=sampling_steps,
                generator=generator,
            )
            for name, value in sparse_diagnostic_totals(
                diagnostics,
                latent.shape[0],
            ).items():
                diagnostic_totals[name] = (
                    value
                    if name not in diagnostic_totals
                    else diagnostic_totals[name] + value
                )
            latent = project_conditioned_latent(
                latent,
                batch["target_latent"],
                batch["constraints"],
            )
            boxes = decode_boxes_batch(latent, batch["areas"])
            terms = batch_physics_terms(
                boxes,
                batch["areas"],
                batch["constraints"],
                batch["b2b"],
                batch["p2b"],
                batch["pins"],
                batch["n_b2b"],
                batch["n_p2b"],
                batch["n_pins"],
            )
            group = group_penalty_batch(
                boxes,
                batch["constraints"][..., 3],
                batch["areas"],
                workers=4,
            )
            aligned = token_ot_reorder_batch(
                latent,
                batch["target_latent"],
                case_indices=count_indices[start : start + batch_size],
                cache=alignment_cache,
            ).to(device)
            target_aspect = batch["target_latent"][..., 2]
            target_aspect_std = target_aspect.std(dim=1, correction=0).clamp_min(
                1.0e-6
            )
            values = {
                "latent_mse": (latent - aligned).pow(2).mean(dim=(1, 2)),
                "rho_mae": (latent[..., 2] - aligned[..., 2]).abs().mean(dim=1),
                "aspect_diversity_ratio": (
                    latent[..., 2].std(dim=1, correction=0) / target_aspect_std
                ),
                "overlap": terms["overlap"],
                "group": group,
                "mib": terms["mib"],
                "boundary": terms["boundary"],
                "wl": terms["wl"],
                "area": terms["area"],
            }
            values["total"] = 0.01 * sum(
                values[name]
                for name in ("overlap", "group", "mib", "boundary", "wl", "area")
            )
            names = tuple(values)
            batch_totals = torch.stack(
                [values[name] for name in names],
                dim=1,
            ).sum(dim=0)
            if metric_totals is None:
                metric_names = names
                metric_totals = batch_totals
            else:
                if names != metric_names:
                    raise RuntimeError("evaluation metric schema changed mid-run")
                metric_totals = metric_totals + batch_totals
            cases += latent.shape[0]
    if was_training:
        model.train()
    resolved_names = tuple(diagnostic_totals)
    resolved_totals = []
    if metric_totals is not None and metric_names is not None:
        resolved_names = (*metric_names, *resolved_names)
        resolved_totals.append(metric_totals)
    if diagnostic_totals:
        resolved_totals.append(
            torch.stack([diagnostic_totals[name] for name in diagnostic_totals])
        )
    if not resolved_totals:
        return {}
    values = torch.cat(resolved_totals).detach().cpu().tolist()
    return {
        name: value / cases
        for name, value in zip(resolved_names, values, strict=True)
    }


__all__ = [
    "augment_geometry",
    "evaluate",
    "sample_batch",
    "select_curve_indices",
    "validate_disjoint_splits",
]

"""Independent atom-level score heads for chip candidate ranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from floorflow.chip_flow.candidate_structural_selector import SELECTOR_METRICS


@dataclass(frozen=True)
class AtomSpec:
    """One deployable metric atom and its selector-label composition."""

    name: str
    metric_weights: tuple[tuple[str, float], ...]

    def validate(self) -> None:
        if not self.name or not self.metric_weights:
            raise ValueError("atom specs require a name and metric weights")
        unknown = {
            metric
            for metric, _weight in self.metric_weights
            if metric not in SELECTOR_METRICS
        }
        if unknown:
            raise ValueError(f"unknown selector metrics: {sorted(unknown)}")
        if any(weight <= 0.0 for _metric, weight in self.metric_weights):
            raise ValueError("atom metric weights must be positive")


ATOM_SPECS: tuple[AtomSpec, ...] = (
    AtomSpec("grouping", (("legalized_grouping", 1.0),)),
    AtomSpec(
        "area_rho",
        (("legalized_area_gap", 1.0), ("legalized_rho_mae", 1.0)),
    ),
    AtomSpec("b2b", (("legalized_b2b_hpwl_gap", 1.0),)),
    AtomSpec("boundary", (("legalized_boundary", 1.0),)),
    AtomSpec("p2b", (("legalized_p2b_hpwl_gap", 1.0),)),
    AtomSpec(
        "legality_tail",
        (
            ("legalized_hard_cost", 1.0),
            ("legalizer_rung", 0.25),
            ("legalizer_solves", 0.25),
        ),
    ),
)
ATOM_NAMES = tuple(spec.name for spec in ATOM_SPECS)


@dataclass(frozen=True)
class AtomHeadOutput:
    """Per-atom utilities, masks and an optional fused candidate utility."""

    utilities: torch.Tensor
    log_variance: torch.Tensor
    reliability: torch.Tensor
    atom_mask: torch.Tensor
    aggregate_utility: torch.Tensor


def _metric_tensor(
    metrics: Mapping[str, float] | torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(metrics, Mapping):
        values = []
        present = []
        for name in SELECTOR_METRICS:
            value = metrics.get(name)
            present.append(value is not None)
            values.append(float(value) if value is not None else 0.0)
        return (
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(present, dtype=torch.bool),
        )
    tensor = torch.as_tensor(metrics, dtype=torch.float32)
    if tensor.shape[-1] != len(SELECTOR_METRICS):
        raise ValueError("metric tensor width must match SELECTOR_METRICS")
    return tensor, torch.ones_like(tensor, dtype=torch.bool)


def selector_metrics_to_atom_targets(
    metrics: Mapping[str, float] | torch.Tensor | np.ndarray,
    metric_mask: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map legalized selector metrics to six lower-is-better atom targets.

    Missing component metrics produce a false atom mask instead of an implicit
    zero target. Tensor inputs use ``SELECTOR_METRICS`` column order.
    """

    tensor, inferred_mask = _metric_tensor(metrics)
    if metric_mask is not None:
        supplied = torch.as_tensor(metric_mask, dtype=torch.bool)
        if supplied.shape != tensor.shape:
            raise ValueError("metric mask shape must match metric tensor")
        inferred_mask = supplied
    targets = []
    masks = []
    for spec in ATOM_SPECS:
        spec.validate()
        value = torch.zeros_like(tensor[..., 0])
        available = torch.ones_like(inferred_mask[..., 0])
        for metric, weight in spec.metric_weights:
            index = SELECTOR_METRICS.index(metric)
            metric_available = inferred_mask[..., index] & torch.isfinite(
                tensor[..., index]
            )
            value = value + torch.where(
                metric_available,
                tensor[..., index] * weight,
                torch.zeros_like(value),
            )
            available = available & metric_available
        targets.append(torch.where(available, value, torch.zeros_like(value)))
        masks.append(available)
    return torch.stack(targets, dim=-1), torch.stack(masks, dim=-1)


def masked_atom_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    log_variance: torch.Tensor | None = None,
    beta: float = 0.1,
) -> torch.Tensor:
    """Masked Smooth-L1 regression with optional heteroscedastic weighting."""

    if predictions.shape != targets.shape or predictions.shape != atom_mask.shape:
        raise ValueError("atom predictions, targets and masks must share a shape")
    mask = torch.as_tensor(
        atom_mask,
        dtype=torch.bool,
        device=predictions.device,
    )
    target_values = torch.as_tensor(
        targets,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    if not bool(mask.any()):
        return predictions.sum() * 0.0
    residual = F.smooth_l1_loss(
        predictions.float(),
        target_values.float(),
        reduction="none",
        beta=float(beta),
    )
    if log_variance is not None:
        if log_variance.shape != predictions.shape:
            raise ValueError("atom log variance shape mismatch")
        variance = torch.as_tensor(
            log_variance,
            dtype=predictions.dtype,
            device=predictions.device,
        )
        residual = 0.5 * (
            residual * torch.exp(-variance.float()).clamp(1.0e-4, 1.0e4)
            + variance.float()
        )
    return residual[mask].mean()


def masked_atom_rank_loss(
    utilities: torch.Tensor,
    targets: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Listwise rank loss for ``(..., candidate, atom)`` utilities."""

    if utilities.shape != targets.shape or utilities.shape != atom_mask.shape:
        raise ValueError("atom utilities, targets and masks must share a shape")
    if utilities.ndim < 2:
        raise ValueError("atom rank loss requires a candidate dimension")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if candidate_mask is not None:
        mask = torch.as_tensor(
            candidate_mask,
            dtype=torch.bool,
            device=utilities.device,
        )
        if mask.shape != utilities.shape[:-1]:
            raise ValueError("candidate mask shape mismatch")
    else:
        mask = torch.ones(
            utilities.shape[:-1],
            dtype=torch.bool,
            device=utilities.device,
        )
    losses = []
    flat_utilities = utilities.reshape(-1, utilities.shape[-2], utilities.shape[-1])
    flat_targets = torch.as_tensor(
        targets,
        dtype=flat_utilities.dtype,
        device=flat_utilities.device,
    ).reshape_as(flat_utilities)
    flat_mask = torch.as_tensor(
        atom_mask,
        dtype=torch.bool,
        device=utilities.device,
    ).reshape_as(flat_utilities)
    flat_candidates = mask.reshape(-1, mask.shape[-1])
    for group in range(flat_utilities.shape[0]):
        for atom in range(flat_utilities.shape[-1]):
            valid = flat_candidates[group] & flat_mask[group, :, atom]
            if int(valid.sum()) < 2:
                continue
            target = torch.softmax(
                -flat_targets[group, valid, atom].detach() / float(temperature),
                dim=-1,
            )
            log_prob = torch.log_softmax(
                -flat_utilities[group, valid, atom],
                dim=-1,
            )
            losses.append(-(target * log_prob).sum())
    if not losses:
        return utilities.sum() * 0.0
    return torch.stack(losses).mean()


class AtomHeadBank(nn.Module):
    """One small independent head per atom over shared candidate evidence."""

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden: int = 32,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden <= 0 or context_dim < 0:
            raise ValueError("invalid atom head dimensions")
        self.feature_dim = int(feature_dim)
        self.context_dim = int(context_dim)
        input_dim = self.feature_dim + self.context_dim
        self.heads = nn.ModuleDict(
            {
                spec.name: nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 2),
                )
                for spec in ATOM_SPECS
            }
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.enabled = False

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def set_trainable_atoms(
        self,
        names: str | tuple[str, ...] | list[str],
    ) -> None:
        """Freeze all but the selected atom heads for independent training."""

        selected = (names,) if isinstance(names, str) else tuple(names)
        unknown = set(selected) - set(ATOM_NAMES)
        if unknown or not selected:
            raise ValueError(f"unknown or empty atom selection: {sorted(unknown)}")
        for name, head in self.heads.items():
            for parameter in head.parameters():
                parameter.requires_grad_(name in selected)
        self.gate.requires_grad_(False)

    def forward(
        self,
        features: torch.Tensor,
        *,
        atom_context: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
        legacy_utility: torch.Tensor | None = None,
    ) -> AtomHeadOutput:
        if features.ndim < 1 or features.shape[-1] != self.feature_dim:
            raise ValueError("atom feature width mismatch")
        leading = features.shape[:-1]
        if candidate_mask is None:
            candidate_mask = torch.ones(
                leading,
                dtype=torch.bool,
                device=features.device,
            )
        candidate_mask = torch.as_tensor(
            candidate_mask,
            dtype=torch.bool,
            device=features.device,
        )
        if candidate_mask.shape != leading:
            raise ValueError("candidate mask shape mismatch")
        safe = torch.where(
            candidate_mask[..., None],
            features.float(),
            torch.zeros((), dtype=torch.float32, device=features.device),
        )
        if self.context_dim:
            if atom_context is None or atom_context.shape != (
                *leading,
                len(ATOM_SPECS),
                self.context_dim,
            ):
                raise ValueError("atom context shape mismatch")
            context = atom_context.float()
        else:
            context = torch.zeros(
                (*leading, len(ATOM_SPECS), 0),
                dtype=torch.float32,
                device=features.device,
            )
        outputs = []
        for index, spec in enumerate(ATOM_SPECS):
            head_input = torch.cat(
                [safe, context[..., index, :]],
                dim=-1,
            )
            outputs.append(self.heads[spec.name](head_input))
        raw = torch.stack(outputs, dim=-2)
        utilities = raw[..., 0]
        log_variance = raw[..., 1].clamp(-8.0, 8.0)
        reliability = torch.exp(-log_variance)
        if atom_mask is None:
            atom_mask = torch.ones(
                (*leading, len(ATOM_SPECS)),
                dtype=torch.bool,
                device=features.device,
            )
        atom_mask = torch.as_tensor(
            atom_mask,
            dtype=torch.bool,
            device=features.device,
        )
        if atom_mask.shape != utilities.shape:
            raise ValueError("atom mask shape mismatch")
        weights = reliability * atom_mask
        pooled = (utilities * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(
            1.0e-6
        )
        base = (
            legacy_utility.float()
            if legacy_utility is not None
            else torch.zeros_like(pooled)
        )
        aggregate = base + self.gate * (pooled - base)
        if not self.enabled:
            aggregate = base
        return AtomHeadOutput(
            utilities=utilities,
            log_variance=log_variance,
            reliability=reliability,
            atom_mask=atom_mask,
            aggregate_utility=aggregate,
        )


def atom_head_loss(
    output: AtomHeadOutput,
    targets: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    rank_weight: float = 0.1,
    temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute masked regression and ranking terms with stable zero fallback."""

    regression = masked_atom_regression_loss(
        output.utilities,
        targets,
        atom_mask,
        log_variance=output.log_variance,
    )
    ranking = masked_atom_rank_loss(
        output.utilities,
        targets,
        atom_mask,
        candidate_mask=candidate_mask,
        temperature=temperature,
    )
    return {
        "atom_total": regression + float(rank_weight) * ranking,
        "atom_regression": regression,
        "atom_rank": ranking,
    }


__all__ = [
    "ATOM_NAMES",
    "ATOM_SPECS",
    "AtomHeadBank",
    "AtomHeadOutput",
    "AtomSpec",
    "atom_head_loss",
    "masked_atom_rank_loss",
    "masked_atom_regression_loss",
    "selector_metrics_to_atom_targets",
]

"""Reusable batched tensors and look-ahead targets for interaction policies."""

from __future__ import annotations

import torch

from floorflow.data.constraint_action_dataset import ACTION_NAMES
from floorflow.data.interaction_candidate_dataset import index_sequences
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.constraint_action_runtime import ActionCandidate
from floorflow.model.constraint_aware_gate import ConstraintAwareGate
from floorflow.model.constraint_state_encoder import room_family_ids


def candidate_from_row(row: dict, operation: str) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=row["candidate_id"],
        role=row["role"],
        node_features=torch.tensor(row["node_features"]),
        adjacency=torch.tensor(row["adjacency"]),
        global_features=torch.tensor(row["global_features"]),
        constraint_diagnostics=torch.tensor(row["constraint_diagnostics"]),
        action_features=torch.tensor(row["action_features"]),
        hard_mask=torch.tensor(
            list(row["hard_mask"].values()),
            dtype=torch.bool,
        ),
        declared_legal=bool(row["hard_feasible"]),
        operation_id=ACTION_NAMES.index(operation),
        utility=float(row["quality"]["utility"]),
        sequence_features=(
            torch.tensor(row["sequence_features"])
            if "sequence_features" in row
            else None
        ),
    )


def batch_tensors(
    groups: list[dict],
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
    *,
    state_blind: bool,
    use_sequence_context: bool = True,
    utility_overrides: dict[str, float] | None = None,
) -> tuple[torch.Tensor, ...]:
    candidates = [
        candidate_from_row(row, group["operation"])
        for group in groups
        for row in group["candidates"]
    ]
    count = len(groups[0]["candidates"])
    if any(len(group["candidates"]) != count for group in groups):
        raise ValueError("candidate counts differ across groups")
    nodes = torch.stack([row.node_features for row in candidates]).to(device)
    adjacency = torch.stack([row.adjacency for row in candidates]).to(device)
    context = torch.cat(
        [
            torch.stack([row.global_features for row in candidates]),
            torch.stack([row.constraint_diagnostics for row in candidates]),
        ],
        dim=-1,
    ).to(device)
    action = torch.stack([row.action_features for row in candidates]).to(device)
    hard = torch.stack([row.hard_mask for row in candidates]).to(device)
    declared = torch.tensor(
        [row.declared_legal for row in candidates],
        device=device,
        dtype=torch.bool,
    )
    hard = hard & declared[:, None]
    if state_blind:
        nodes = torch.zeros_like(nodes)
        adjacency = torch.zeros_like(adjacency)
        context = torch.zeros_like(context)
    sequence = torch.stack(
        [
            (
                row.sequence_features
                if row.sequence_features is not None
                else torch.zeros(12)
            )
            for row in candidates
        ]
    ).to(device)
    if state_blind or not use_sequence_context:
        sequence = torch.zeros_like(sequence)
    operations = torch.tensor(
        [row.operation_id for row in candidates],
        device=device,
    )
    family = room_family_ids(taxonomy).to(device)[None].expand(
        len(candidates),
        -1,
    )
    utility = torch.tensor(
        [
            (
                utility_overrides.get(row.candidate_id, row.utility)
                if utility_overrides is not None
                else row.utility
            )
            for row in candidates
        ],
        device=device,
    ).reshape(len(groups), count)
    legal = torch.tensor(
        [row.role != "near_violation" for row in candidates],
        device=device,
    ).reshape(len(groups), count)
    return (
        nodes,
        adjacency,
        family,
        operations,
        action,
        hard,
        context,
        utility,
        legal,
        sequence,
    )


def forward_groups(
    model: ConstraintAwareGate,
    groups: list[dict],
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
    *,
    state_blind: bool,
    use_sequence_context: bool = True,
    utility_overrides: dict[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = batch_tensors(
        groups,
        taxonomy,
        device,
        state_blind=state_blind,
        use_sequence_context=use_sequence_context,
        utility_overrides=utility_overrides,
    )
    (
        nodes,
        adjacency,
        family,
        operations,
        action,
        hard,
        context,
        utility,
        legal,
        sequence,
    ) = tensors
    kwargs = {}
    if hasattr(model, "sequence_trunk") and use_sequence_context and not state_blind:
        kwargs["sequence_features"] = sequence
    output = model(
        nodes,
        adjacency,
        family,
        operations,
        action,
        hard,
        global_features=context,
        **kwargs,
    )
    count = len(groups[0]["candidates"])
    for key in (
        "gate_logits",
        "gate_probability",
        "legal_logits",
        "uncertainty",
        "hard_feasible",
        "interaction_value",
    ):
        if key in output:
            output[key] = output[key].reshape(len(groups), count)
    hard = hard.reshape(len(groups), count, -1)
    return output, utility, hard, legal


def normalized_utility(
    utility: torch.Tensor,
    feasible: torch.Tensor,
) -> torch.Tensor:
    masked = utility.masked_fill(~feasible, torch.nan)
    mean = masked.nanmean(dim=-1, keepdim=True)
    centered = utility - mean
    variance = centered.masked_fill(~feasible, torch.nan).pow(2).nanmean(
        dim=-1,
        keepdim=True,
    )
    return centered / variance.sqrt().clamp_min(1.0e-6)


def lookahead_utility_map(groups: list[dict]) -> dict[str, float]:
    sequences = index_sequences(groups)
    values = {}
    for sequence in sequences.values():
        for step in sorted(sequence["groups"], reverse=True):
            group = sequence["groups"][step]
            for candidate in group["candidates"]:
                value = float(candidate["quality"]["utility"])
                child = sequence["children"].get(candidate["candidate_id"])
                if child is not None:
                    child_values = [
                        float(row["quality"]["utility"])
                        for row in child["candidates"]
                        if row["role"] != "near_violation"
                    ]
                    if child_values:
                        value = max(child_values)
                values[candidate["candidate_id"]] = value
    return values


__all__ = [
    "batch_tensors",
    "candidate_from_row",
    "forward_groups",
    "lookahead_utility_map",
    "normalized_utility",
]

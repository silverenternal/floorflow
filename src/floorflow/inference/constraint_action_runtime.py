"""Hard-masked single-trajectory action selection for FloorFlow."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.constraint_aware_gate import ConstraintAwareGate
from floorflow.model.constraint_state_encoder import room_family_ids


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    role: str
    node_features: torch.Tensor
    adjacency: torch.Tensor
    global_features: torch.Tensor
    constraint_diagnostics: torch.Tensor
    action_features: torch.Tensor
    hard_mask: torch.Tensor
    declared_legal: bool
    operation_id: int
    utility: float
    sequence_features: torch.Tensor | None = None


@dataclass(frozen=True)
class ActionDecision:
    index: int
    candidate_id: str
    score: float
    legal_probability: float
    uncertainty: float
    hard_feasible: bool
    selected_role: str
    selection_source: str
    invalid_probability_mass: float


def _candidate_hash(candidate_id: str) -> int:
    return int(
        hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:15],
        16,
    )


class ConstraintActionRuntime:
    """Choose one action from legal alternatives without complete-plan best-of-N."""

    def __init__(
        self,
        model: ConstraintAwareGate,
        *,
        alpha: float = 0.0,
        legal_fallback_threshold: float = 0.0,
        state_blind: bool = False,
        sequence_blind: bool = False,
    ) -> None:
        self.model = model.eval()
        self.alpha = float(alpha)
        self.legal_fallback_threshold = float(legal_fallback_threshold)
        self.state_blind = bool(state_blind)
        self.sequence_blind = bool(sequence_blind)

    def _model_outputs(
        self,
        candidates: list[ActionCandidate],
        taxonomy: FloorplanTaxonomy,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        node_features = torch.stack(
            [candidate.node_features for candidate in candidates]
        ).to(device)
        adjacency = torch.stack(
            [candidate.adjacency for candidate in candidates]
        ).to(device)
        global_features = torch.cat(
            [
                torch.stack(
                    [candidate.global_features for candidate in candidates]
                ),
                torch.stack(
                    [candidate.constraint_diagnostics for candidate in candidates]
                ),
            ],
            dim=-1,
        ).to(device)
        if self.state_blind:
            node_features = torch.zeros_like(node_features)
            adjacency = torch.zeros_like(adjacency)
            global_features = torch.zeros_like(global_features)
        action_features = torch.stack(
            [candidate.action_features for candidate in candidates]
        ).to(device)
        hard_mask = torch.stack(
            [candidate.hard_mask for candidate in candidates]
        ).to(device)
        operation_ids = torch.tensor(
            [candidate.operation_id for candidate in candidates],
            device=device,
        )
        family_ids = room_family_ids(taxonomy).to(device)
        family_ids = family_ids[None].expand(len(candidates), -1)
        sequence = None
        if any(candidate.sequence_features is not None for candidate in candidates):
            sequence = torch.stack(
                [
                    (
                        candidate.sequence_features
                        if candidate.sequence_features is not None
                        else torch.zeros(12)
                    )
                    for candidate in candidates
                ]
            ).to(device)
            if self.state_blind or self.sequence_blind:
                sequence = torch.zeros_like(sequence)
        kwargs = {}
        if (
            hasattr(self.model, "sequence_trunk")
            and not self.state_blind
            and not self.sequence_blind
        ):
            kwargs["sequence_features"] = sequence
        return self.model(
            node_features,
            adjacency,
            family_ids,
            operation_ids,
            action_features,
            hard_mask,
            global_features=global_features,
            **kwargs,
        )

    def select(
        self,
        candidates: list[ActionCandidate],
        taxonomy: FloorplanTaxonomy,
        *,
        device: torch.device | None = None,
    ) -> ActionDecision:
        if not candidates:
            raise ValueError("action selection requires candidates")
        device = device or next(self.model.parameters()).device
        with torch.no_grad():
            output = self._model_outputs(candidates, taxonomy, device)
            declared = torch.tensor(
                [candidate.declared_legal for candidate in candidates],
                device=device,
                dtype=torch.bool,
            )
            feasible = output["hard_feasible"].bool() & declared
            effective_probability = output["gate_probability"] * declared
            if not bool(feasible.any()):
                index = max(
                    range(len(candidates)),
                    key=lambda item: _candidate_hash(candidates[item].candidate_id),
                )
                return ActionDecision(
                    index,
                    candidates[index].candidate_id,
                    0.0,
                    0.0,
                    float(output["uncertainty"][index]),
                    False,
                    candidates[index].role,
                    "invalid_fallback",
                    0.0,
                )
            scores = (
                output["gate_logits"]
                + self.alpha * output["legal_logits"]
            )
            scores = scores.masked_fill(~feasible, -torch.inf)
            index = int(scores.argmax())
            legal_probability = float(
                output["legal_logits"][index].sigmoid()
            )
            source = "cagn"
            if legal_probability < self.legal_fallback_threshold:
                legal_scores = output["legal_logits"].masked_fill(
                    ~feasible,
                    -torch.inf,
                )
                index = int(legal_scores.argmax())
                source = "legal_fallback"
            return ActionDecision(
                index=index,
                candidate_id=candidates[index].candidate_id,
                score=float(scores[index]),
                legal_probability=float(
                    output["legal_logits"][index].sigmoid()
                ),
                uncertainty=float(output["uncertainty"][index]),
                hard_feasible=bool(feasible[index]),
                selected_role=candidates[index].role,
                selection_source=source,
                invalid_probability_mass=float(
                    effective_probability[~feasible].sum()
                ),
            )


def hard_rule_selection(
    candidates: list[ActionCandidate],
) -> ActionDecision:
    """Matched hard-rule baseline with deterministic candidate tie-breaking."""

    feasible = [
        candidate
        for candidate in candidates
        if bool(candidate.hard_mask.all()) and candidate.declared_legal
    ]
    pool = feasible or candidates
    selected = max(pool, key=lambda item: _candidate_hash(item.candidate_id))
    return ActionDecision(
        index=candidates.index(selected),
        candidate_id=selected.candidate_id,
        score=1.0 if feasible else 0.0,
        legal_probability=(
            1.0
            if bool(selected.hard_mask.all()) and selected.declared_legal
            else 0.0
        ),
        uncertainty=1.0,
        hard_feasible=(
            bool(selected.hard_mask.all()) and selected.declared_legal
        ),
        selected_role=selected.role,
        selection_source="hard_rule_hash",
        invalid_probability_mass=0.0,
    )


__all__ = [
    "ActionCandidate",
    "ActionDecision",
    "ConstraintActionRuntime",
    "hard_rule_selection",
]

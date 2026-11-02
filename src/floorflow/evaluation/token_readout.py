"""Token-branch readout metrics for training and validation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.heads.floorplan_token_head import FloorplanTokenHead
from floorflow.inference.floorplan_token_generation import (
    TokenGenerationBudgetError,
    generate_floorplan_tokens,
)
from floorflow.objectives.token_grammar import (
    TOKEN_STAGES,
    FloorplanTokenVocabulary,
    grammar_masks_for_ids,
)
from floorflow.representation.unified_floorplan_tokens import FloorplanToken


def token_teacher_forced_metrics(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid: torch.Tensor,
    stage_ids: torch.Tensor,
    vocabulary: FloorplanTokenVocabulary,
    grammar_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("token logits and targets must align")
    if valid.shape != target_ids.shape or stage_ids.shape != target_ids.shape:
        raise ValueError("token masks and stage ids must align")
    with torch.no_grad():
        kind_names = tuple(sorted({*vocabulary.kind_names, "UNK"}))
        kind_index = {kind: index for index, kind in enumerate(kind_names)}
        token_kind_values = [
            kind_index[value.split("|", 1)[0]]
            for value in vocabulary.tokens
        ]
        token_kind_values.append(kind_index["UNK"])
        token_kind = torch.tensor(
            token_kind_values,
            dtype=torch.long,
            device=logits.device,
        )
        predicted = logits.argmax(dim=-1)
        selected_targets = target_ids[valid]
        selected_predicted = predicted[valid]
        metrics = {
            "token_accuracy": float(
                (
                    (selected_targets == selected_predicted).float().sum()
                    / selected_targets.numel()
                )
                if selected_targets.numel()
                else logits.new_tensor(0.0)
            ),
        }
        safe_targets = target_ids.long().clamp(0, token_kind.numel() - 1)
        target_kind_ids = token_kind[safe_targets]
        predicted_kind_ids = token_kind[predicted.long()]
        selected_target_kinds = target_kind_ids[valid]
        selected_predicted_kinds = predicted_kind_ids[valid]
        number_of_kinds = len(kind_names)
        confusion = torch.bincount(
            selected_target_kinds * number_of_kinds
            + selected_predicted_kinds,
            minlength=number_of_kinds * number_of_kinds,
        ).reshape(number_of_kinds, number_of_kinds)
        true_positive = confusion.diag()
        false_positive = confusion.sum(dim=0) - true_positive
        false_negative = confusion.sum(dim=1) - true_positive
        precision = true_positive / (
            true_positive + false_positive
        ).clamp_min(1)
        recall = true_positive / (
            true_positive + false_negative
        ).clamp_min(1)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(
            1.0e-6
        )
        present = (confusion.sum(dim=0) + confusion.sum(dim=1)) > 0
        for index, kind in enumerate(kind_names):
            metrics[f"token_kind_f1_{kind.lower()}"] = float(f1[index])
        metrics["token_macro_f1"] = float(
            (f1 * present.to(f1.dtype)).sum()
            / present.sum().clamp_min(1)
        )
        stage_losses = []
        stage_present = []
        for stage_id, stage_name in enumerate(TOKEN_STAGES):
            selected = valid & (stage_ids == stage_id)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                safe_targets.reshape(-1),
                reduction="none",
            ).reshape_as(target_ids)
            weights = selected.to(losses.dtype)
            stage_loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
            metrics[f"token_stage_ce_{stage_name}"] = float(stage_loss)
            stage_losses.append(stage_loss)
            stage_present.append((selected.sum() > 0).to(stage_loss.dtype))
        stage_present_tensor = torch.stack(stage_present)
        metrics["token_stage_ce_mean"] = float(
            (torch.stack(stage_losses) * stage_present_tensor).sum()
            / stage_present_tensor.sum().clamp_min(1.0)
        )
        if grammar_mask is None:
            grammar_mask = grammar_masks_for_ids(
                target_ids,
                vocabulary,
                valid,
            )
        if grammar_mask.shape[:2] != logits.shape[:2]:
            raise ValueError("grammar mask positions must match token logits")
        if grammar_mask.shape[-1] < logits.shape[-1]:
            raise ValueError("grammar mask vocabulary is smaller than logits")
        grammar_mask = grammar_mask[..., : logits.shape[-1]]
        predicted_allowed = grammar_mask.gather(
            -1,
            predicted.clamp(0, grammar_mask.shape[-1] - 1)[..., None],
        ).squeeze(-1)
        metrics["token_grammar_invalid_rate"] = float(
            (~predicted_allowed & valid).sum().float() / valid.sum().clamp_min(1)
        )
        for name, kind in (
            ("room", "ROOM_BEGIN"),
            ("wall", "WALL"),
            ("opening", "OPENING"),
        ):
            kind_id = kind_index.get(kind)
            expected = (
                (selected_target_kinds == kind_id).sum()
                if kind_id is not None
                else selected_target_kinds.new_tensor(0)
            )
            actual = (
                (selected_predicted_kinds == kind_id).sum()
                if kind_id is not None
                else selected_predicted_kinds.new_tensor(0)
            )
            metrics[f"token_{name}_count_abs_error"] = float(
                abs(expected - actual)
            )
        end_id = kind_index["END"]
        end_valid = valid & (target_kind_ids == end_id)
        metrics["token_teacher_forced_end_accuracy"] = float(
            ((predicted_kind_ids == end_id) & end_valid).sum()
            / end_valid.sum().clamp_min(1)
        )
    return metrics


@torch.no_grad()
def constrained_generation_metrics(
    head: FloorplanTokenHead,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    vocabulary: FloorplanTokenVocabulary,
    target_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    efficiency_budget: int,
) -> dict[str, float]:
    prompt_length = min(5, int(target_ids.shape[0]))
    prompt = []
    for token_id in target_ids[:prompt_length].detach().cpu().tolist():
        value = vocabulary.decode(int(token_id))
        if value.startswith("<"):
            return {"token_generation_completed": 0.0, "token_generated_count": 0.0}
        prompt.append(FloorplanToken.parse(value))
    try:
        tokens, _scores = generate_floorplan_tokens(
            head,
            context,
            context_mask,
            vocabulary,
            prompt,
            max_new_tokens=max_new_tokens,
        )
    except TokenGenerationBudgetError as error:
        return {
            "token_generation_completed": 0.0,
            "token_generated_count": float(
                max(0, len(error.tokens) - prompt_length)
            ),
            "token_generation_within_efficiency_budget": 0.0,
        }
    except ValueError:
        return {"token_generation_completed": 0.0, "token_generated_count": 0.0}
    return {
        "token_generation_completed": float(tokens[-1].kind == "END"),
        "token_generated_count": float(max(0, len(tokens) - prompt_length)),
        "token_generation_within_efficiency_budget": float(
            tokens[-1].kind == "END"
            and len(tokens) - prompt_length <= int(efficiency_budget)
        ),
    }


__all__ = [
    "constrained_generation_metrics",
    "token_teacher_forced_metrics",
]

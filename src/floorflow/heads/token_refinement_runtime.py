"""Runtime composition for token hidden states and vector refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.heads.floorplan_token_head import FloorplanTokenHead
from floorflow.heads.token_vector_refiner import TokenVectorRefiner
from floorflow.objectives.token_grammar import TOKEN_STAGES


def apply_token_refinement(
    token_head: FloorplanTokenHead,
    token_refiner: TokenVectorRefiner,
    output: dict[str, torch.Tensor],
    token_ids: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    token_stage_ids: torch.Tensor | None = None,
    token_input_ids: torch.Tensor | None = None,
    token_rollout_steps: int = 0,
) -> dict[str, torch.Tensor]:
    if "token_context_tokens" not in output:
        raise ValueError("base output is missing token context")
    mask = (
        token_mask
        if token_mask is not None
        else torch.ones_like(token_ids, dtype=torch.bool)
    )
    token_stage_ids = (
        torch.zeros_like(token_ids)
        if token_stage_ids is None
        else token_stage_ids
    )
    head_input_ids = token_input_ids if token_input_ids is not None else token_ids
    hidden = None
    token_logits = None
    for step in range(max(0, int(token_rollout_steps)) + 1):
        hidden, token_logits = token_head.forward_with_hidden(
            output["token_context_tokens"],
            head_input_ids,
            mask,
            token_stage_ids,
            output["token_context_valid"],
        )
        if step >= int(token_rollout_steps):
            break
        predicted = token_logits.argmax(dim=-1)
        rolled = head_input_ids.clone()
        prefix_length = min(5, rolled.shape[1])
        if rolled.shape[1] > prefix_length:
            rolled[:, prefix_length:] = predicted[:, prefix_length - 1 : -1]
        head_input_ids = torch.where(mask, rolled, head_input_ids)
    if hidden is None or token_logits is None:
        raise RuntimeError("token head did not execute")
    target_ids = head_input_ids.clamp(0, token_head.vocab_size - 1)
    token_probability = token_logits.log_softmax(dim=-1).gather(
        -1,
        target_ids[..., None],
    ).squeeze(-1).exp()
    confidence = (
        token_probability * mask
    ).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    stage_one_hot = F.one_hot(
        token_stage_ids.long(),
        num_classes=len(TOKEN_STAGES),
    ).to(token_probability.dtype)
    stage_mask = stage_one_hot * mask[..., None]
    stage_confidence = (
        (token_probability[..., None] * stage_mask).sum(dim=1)
        / stage_mask.sum(dim=1).clamp_min(1.0)
    )
    refined = token_refiner(
        output,
        hidden,
        mask,
        confidence,
        stage_confidence,
    )
    refined.update(
        {
            "token_logits": token_logits,
            "token_ids": token_ids,
            "token_refinement_hidden": hidden,
        }
    )
    return refined


__all__ = ["apply_token_refinement"]

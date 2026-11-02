"""Grammar-constrained generation through the FloorFlow token head."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from floorflow.heads.floorplan_token_head import FloorplanTokenHead
from floorflow.objectives.token_grammar import (
    STAGE_INDEX,
    FloorplanGrammarState,
    FloorplanTokenVocabulary,
    stage_for_prefix,
    stage_ids_for_tokens,
)
from floorflow.representation.unified_floorplan_tokens import FloorplanToken


class TokenGenerationBudgetError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        tokens: list[FloorplanToken],
        scores: list[float],
    ) -> None:
        super().__init__(message)
        self.tokens = tokens
        self.scores = scores


@torch.no_grad()
def generate_floorplan_tokens(
    head: FloorplanTokenHead,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    vocabulary: FloorplanTokenVocabulary,
    prompt: Sequence[FloorplanToken],
    *,
    max_new_tokens: int,
    immutable_reference: Sequence[FloorplanToken] | None = None,
    immutable_mask: Sequence[bool] | None = None,
) -> tuple[list[FloorplanToken], list[float]]:
    """Generate one canonical token sequence under the strict grammar."""

    if context.shape[0] != 1:
        raise ValueError("constrained token generation currently supports batch size one")
    if not prompt:
        raise ValueError("token generation requires a prompt")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prefix = list(prompt)
    if prefix[-1].kind == "END":
        return prefix, []
    scores: list[float] = []
    device = context.device
    state = FloorplanGrammarState(vocabulary, prefix)
    target_stages = stage_ids_for_tokens(prefix)
    target_stages.append(STAGE_INDEX[stage_for_prefix(prefix)])
    input_ids = [head.bos_id, *[vocabulary.index(token) for token in prefix]]
    cache = None
    logits = None
    for position, input_id in enumerate(input_ids):
        logits, cache = head.incremental_step(
            context,
            context_mask,
            torch.tensor([input_id], dtype=torch.long, device=device),
            torch.tensor(
                [target_stages[position]],
                dtype=torch.long,
                device=device,
            ),
            cache,
        )
    if logits is None or cache is None:
        raise RuntimeError("failed to initialize token KV cache")
    for _ in range(int(max_new_tokens)):
        allowed = state.allowed_mask(device)
        forced = (
            immutable_reference is not None
            and immutable_mask is not None
            and len(prefix) < len(immutable_mask)
            and len(prefix) < len(immutable_reference)
            and bool(immutable_mask[len(prefix)])
        )
        if forced:
            token = immutable_reference[len(prefix)]
            token_id = vocabulary.index(token)
            constrained = logits[0]
            if not bool(allowed[token_id]):
                raise ValueError("immutable token violates the grammar")
        else:
            if not bool(allowed.any()):
                raise ValueError("grammar has no valid continuation")
            constrained = logits[0].masked_fill(~allowed, -torch.inf)
            token_id = int(constrained.argmax())
        value = vocabulary.decode(token_id)
        if value.startswith("<"):
            raise ValueError(f"generation selected special token: {value}")
        if not forced:
            token = FloorplanToken.parse(value)
        prefix.append(token)
        state.append(token)
        scores.append(float(constrained[token_id]))
        if token.kind == "END":
            break
        next_stage = STAGE_INDEX[stage_for_prefix(prefix)]
        logits, cache = head.incremental_step(
            context,
            context_mask,
            torch.tensor([token_id], dtype=torch.long, device=device),
            torch.tensor([next_stage], dtype=torch.long, device=device),
            cache,
        )
    if prefix[-1].kind != "END":
        raise TokenGenerationBudgetError(
            "token generation reached its budget without END",
            tokens=prefix,
            scores=scores,
        )
    return prefix, scores


__all__ = ["TokenGenerationBudgetError", "generate_floorplan_tokens"]

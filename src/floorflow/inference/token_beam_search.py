"""KV-cached grammar-constrained beam search with verifier reranking."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from floorflow.heads.floorplan_token_head import (
    FloorplanTokenHead,
    FloorplanTokenKVCache,
)
from floorflow.objectives.token_grammar import (
    STAGE_INDEX,
    FloorplanGrammarState,
    FloorplanTokenVocabulary,
    stage_for_prefix,
    stage_ids_for_tokens,
)
from floorflow.representation.unified_floorplan_tokens import FloorplanToken


@torch.no_grad()
def beam_search_floorplan_tokens(
    head: FloorplanTokenHead,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    vocabulary: FloorplanTokenVocabulary,
    prompt: Sequence[FloorplanToken],
    *,
    max_new_tokens: int,
    beam_width: int = 4,
    length_penalty: float = 1.0,
    immutable_reference: Sequence[FloorplanToken] | None = None,
    immutable_mask: Sequence[bool] | None = None,
) -> list[tuple[list[FloorplanToken], float]]:
    if context.shape[0] != 1 or beam_width <= 0:
        raise ValueError("beam search requires batch one and positive beam width")
    prefix = list(prompt)
    state = FloorplanGrammarState(vocabulary, prefix)
    target_stages = stage_ids_for_tokens(prefix)
    target_stages.append(STAGE_INDEX[stage_for_prefix(prefix)])
    inputs = [head.bos_id, *[vocabulary.index(token) for token in prefix]]
    cache = None
    logits = None
    device = context.device
    for position, token_id in enumerate(inputs):
        logits, cache = head.incremental_step(
            context,
            context_mask,
            torch.tensor([token_id], dtype=torch.long, device=device),
            torch.tensor([target_stages[position]], dtype=torch.long, device=device),
            cache,
        )
    if logits is None or cache is None:
        raise RuntimeError("failed to initialize beam cache")
    beams = [(prefix, state, cache, logits, 0.0)]
    completed: list[tuple[list[FloorplanToken], float]] = []
    token_lookup = {
        token_id: FloorplanToken.parse(value)
        for token_id, value in enumerate(vocabulary.tokens)
    }
    for _ in range(int(max_new_tokens)):
        branch_values = []
        branch_ids = []
        parent_indices = []
        parent_stages = []
        parent_scores = []
        for tokens, grammar, beam_cache, beam_logits, score in beams:
            allowed = grammar.allowed_mask(device)
            log_probability = beam_logits[0].log_softmax(dim=-1)
            forced = (
                immutable_reference is not None
                and immutable_mask is not None
                and len(tokens) < len(immutable_mask)
                and len(tokens) < len(immutable_reference)
                and bool(immutable_mask[len(tokens)])
            )
            if forced:
                forced_token = immutable_reference[len(tokens)]
                forced_id = vocabulary.index(forced_token)
                values = log_probability[forced_id].reshape(1)
                indices = torch.tensor(
                    [forced_id],
                    dtype=torch.long,
                    device=device,
                )
            else:
                values, indices = log_probability.masked_fill(
                    ~allowed,
                    -torch.inf,
                ).topk(beam_width)
            branch_values.append(values)
            branch_ids.append(indices)
            parent_indices.append(
                torch.full(
                    (indices.numel(),),
                    len(parent_indices),
                    dtype=torch.long,
                    device=device,
                )
            )
            parent_stages.append(
                torch.full(
                    (indices.numel(),),
                    STAGE_INDEX[stage_for_prefix(tokens)],
                    dtype=torch.long,
                    device=device,
                )
            )
            parent_scores.append(
                beam_logits.new_full((indices.numel(),), float(score))
            )
        if not branch_values:
            break
        flat_values = torch.cat(branch_values)
        flat_ids = torch.cat(branch_ids)
        flat_parent = torch.cat(parent_indices)
        flat_stages = torch.cat(parent_stages)
        flat_scores = torch.cat(parent_scores) + flat_values
        grouped_cache = FloorplanTokenKVCache.stack(
            [beam[2] for beam in beams],
        )
        grouped_cache = grouped_cache.index_select(flat_parent)
        repeated_context = context.expand(flat_ids.numel(), -1, -1)
        repeated_context_mask = context_mask.expand(flat_ids.numel(), -1)
        next_logits, next_cache = head.incremental_step(
            repeated_context,
            repeated_context_mask,
            flat_ids,
            flat_stages,
            grouped_cache,
        )
        selected_scores, selected = torch.topk(
            flat_scores,
            min(beam_width, flat_scores.numel()),
        )
        selected_rows = selected.tolist()
        selected_parents = flat_parent[selected].tolist()
        selected_ids = flat_ids[selected].tolist()
        next_beams = []
        for selected_index, parent_index, token_id, selected_score in zip(
            selected_rows,
            selected_parents,
            selected_ids,
            selected_scores.tolist(),
        ):
            if selected_score == float("-inf"):
                continue
            token = token_lookup.get(token_id)
            if token is None:
                continue
            tokens, grammar = beams[parent_index][:2]
            next_tokens = [*tokens, token]
            if token.kind == "END":
                completed.append((next_tokens, float(selected_score)))
                continue
            next_state = FloorplanGrammarState(vocabulary, tokens)
            next_state.append(token)
            row = torch.tensor(
                [selected_index],
                dtype=torch.long,
                device=device,
            )
            next_beams.append(
                (
                    next_tokens,
                    next_state,
                    next_cache.index_select(row),
                    next_logits[selected_index : selected_index + 1],
                    float(selected_score),
                )
            )
        beams = next_beams
        if not beams:
            break
    return sorted(
        completed,
        key=lambda item: item[1] / max(1, len(item[0])) ** length_penalty,
        reverse=True,
    )


@torch.no_grad()
def verifier_guided_beam_search(
    head: FloorplanTokenHead,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    vocabulary: FloorplanTokenVocabulary,
    prompt: Sequence[FloorplanToken],
    verifier: Callable[[list[FloorplanToken]], float],
    *,
    max_new_tokens: int,
    beam_width: int = 4,
    length_penalty: float = 1.0,
    immutable_reference: Sequence[FloorplanToken] | None = None,
    immutable_mask: Sequence[bool] | None = None,
) -> tuple[list[FloorplanToken], float]:
    candidates = beam_search_floorplan_tokens(
        head,
        context,
        context_mask,
        vocabulary,
        prompt,
        max_new_tokens=max_new_tokens,
        beam_width=beam_width,
        length_penalty=length_penalty,
        immutable_reference=immutable_reference,
        immutable_mask=immutable_mask,
    )
    if not candidates:
        raise ValueError("beam search produced no completed sequence")
    ranked = [
        (
            verifier(tokens),
            score / max(1, len(tokens)) ** length_penalty,
            tokens,
        )
        for tokens, score in candidates
    ]
    best = max(ranked, key=lambda item: (item[0], item[1]))
    return best[2], best[0]


__all__ = ["beam_search_floorplan_tokens", "verifier_guided_beam_search"]

"""Two-pass inference from token generation to token-conditioned vector CAD."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from floorflow.inference.floorplan_token_generation import (
    generate_floorplan_tokens,
)
from floorflow.inference.junction_face_readout import junction_face_candidate
from floorflow.inference.token_beam_search import beam_search_floorplan_tokens
from floorflow.inference.typed_state_machine import select_verified_state
from floorflow.objectives.token_grammar import (
    FloorplanTokenVocabulary,
    stage_ids_for_tokens,
)
from floorflow.representation.unified_floorplan_tokens import FloorplanToken


@torch.no_grad()
def token_refined_generation(
    model,
    boundary_mask: torch.Tensor,
    boundary_valid: torch.Tensor,
    boundary_points: torch.Tensor,
    entrance_points: torch.Tensor,
    entrance_valid: torch.Tensor,
    vocabulary: FloorplanTokenVocabulary,
    prompt: Sequence[FloorplanToken],
    *,
    max_new_tokens: int,
    beam_width: int = 1,
    immutable_reference: Sequence[FloorplanToken] | None = None,
    immutable_mask: Sequence[bool] | None = None,
    verifier: Callable[[dict], float] | None = None,
) -> tuple[
    list[dict[str, torch.Tensor]],
    list[dict[str, torch.Tensor]],
    list[list[FloorplanToken]],
]:
    """Generate tokens first, then refine cached base outputs per sample."""

    base = model(
        boundary_mask,
        boundary_valid,
        boundary_points,
        entrance_points,
        entrance_valid,
    )
    if "token_context_tokens" not in base:
        raise ValueError("model does not expose token context")
    refined_outputs = []
    base_outputs = []
    generated_outputs = []
    for row in range(boundary_mask.shape[0]):
        row_output = {
            key: value[row : row + 1]
            if torch.is_tensor(value) and value.ndim > 0
            and value.shape[0] == boundary_mask.shape[0]
            else value
            for key, value in base.items()
        }
        base_outputs.append(row_output)
        if beam_width > 1:
            candidates = beam_search_floorplan_tokens(
                model.token_head,
                row_output["token_context_tokens"],
                row_output["token_context_valid"],
                vocabulary,
                prompt,
                max_new_tokens=max_new_tokens,
                beam_width=beam_width,
                immutable_reference=immutable_reference,
                immutable_mask=immutable_mask,
            )
            if not candidates:
                raise ValueError("beam search produced no complete sequence")
            generated = candidates[0][0]
        else:
            generated, _scores = generate_floorplan_tokens(
                model.token_head,
                row_output["token_context_tokens"],
                row_output["token_context_valid"],
                vocabulary,
                prompt,
                max_new_tokens=max_new_tokens,
                immutable_reference=immutable_reference,
                immutable_mask=immutable_mask,
            )
        token_ids = torch.tensor(
            [[vocabulary.index(token) for token in generated]],
            dtype=torch.long,
            device=boundary_mask.device,
        )
        stages = torch.tensor(
            [stage_ids_for_tokens(generated)],
            dtype=torch.long,
            device=boundary_mask.device,
        )
        refined_output = model.refine_with_token_ids(
            row_output,
            token_ids,
            torch.ones_like(token_ids, dtype=torch.bool),
            stages,
        )
        if verifier is not None:
            candidates = {
                "vector": row_output,
                "token_refined": refined_output,
            }
            if "codetree_polygons_aligned" in refined_output:
                codetree = dict(refined_output)
                codetree["room_polygons"] = refined_output[
                    "codetree_polygons_aligned"
                ]
                candidates["codetree"] = codetree
            face_candidate = junction_face_candidate(refined_output)
            if face_candidate is not None:
                candidates["junction_faces"] = face_candidate
            refined_output, _report = select_verified_state(
                candidates,
                verifier,
                fallback="vector",
            )
        refined_outputs.append(refined_output)
        generated_outputs.append(generated)
    return base_outputs, refined_outputs, generated_outputs


__all__ = ["token_refined_generation"]

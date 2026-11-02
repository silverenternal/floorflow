"""Token-conditioned typed floorplan state transition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from floorflow.inference.junction_face_readout import junction_face_candidate
from floorflow.representation.typed_floorplan_state import TypedFloorplanState

if TYPE_CHECKING:
    from floorflow.model.three_layer_controller import ThreeLayerFloorplanController


def transition_typed_state_with_tokens(
    controller: ThreeLayerFloorplanController,
    output: dict[str, torch.Tensor],
    token_ids: torch.Tensor,
    token_mask: torch.Tensor | None,
    token_stage_ids: torch.Tensor | None,
    token_input_ids: torch.Tensor | None,
    token_rollout_steps: int,
    *,
    codetree_primary: bool,
    topology_faces_primary: bool,
    allow_hard_primary: bool,
    differentiable_alignment: bool,
) -> tuple[dict[str, torch.Tensor], TypedFloorplanState]:
    refined, state = controller.room_and_geometry_transition(
        output,
        token_ids,
        token_mask,
        token_stage_ids,
        token_input_ids,
        token_rollout_steps,
    )
    mask = (
        token_mask
        if token_mask is not None
        else torch.ones_like(token_ids, dtype=torch.bool)
    )
    stages = (
        token_stage_ids
        if token_stage_ids is not None
        else torch.zeros_like(token_ids)
    )
    refined, state = controller.topology_transition(
        refined,
        state,
        mask,
        stages,
    )
    topology_primary_applied = False
    if topology_faces_primary and allow_hard_primary:
        face_output = junction_face_candidate(refined)
        if face_output is not None:
            base_state = state.clone()
            transformed_face = TypedFloorplanState.from_predictions(face_output)
            transformed_face = transformed_face.align_rooms_to(base_state)
            state = transformed_face
            state.room_features = base_state.room_features
            state.room_instance_ids = base_state.room_instance_ids
            refined.update(face_output)
            refined["room_polygons"] = state.room_polygons
            refined["junction_faces_primary_applied"] = token_ids.new_tensor(1.0)
            topology_primary_applied = True
    refined, state = controller.geometry_transition(
        refined,
        state,
        mask,
        stages,
        codetree_primary=codetree_primary and not topology_primary_applied,
        differentiable_alignment=differentiable_alignment,
    )
    if (
        codetree_primary
        and not topology_primary_applied
        and state.room_polygons is not None
    ):
        refined["codetree_primary_applied"] = token_ids.new_tensor(1.0)
    refined.pop("token_refinement_hidden", None)
    predictions = state.to_predictions()
    refined.update(predictions)
    return refined, state


__all__ = ["transition_typed_state_with_tokens"]

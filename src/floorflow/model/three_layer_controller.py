"""Execution controller for room, topology, and geometry transitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from floorflow.heads.codetree_anchor_decoder import CodeTreeAnchorDecoder
from floorflow.heads.floorplan_token_head import FloorplanTokenHead
from floorflow.heads.token_graph_residual_head import TokenGraphResidualHead
from floorflow.heads.token_refinement_runtime import apply_token_refinement
from floorflow.heads.token_vector_refiner import TokenVectorRefiner
from floorflow.representation.typed_floorplan_state import TypedFloorplanState
from floorflow.representation.typed_stage_layers import stage_layers_dict


@dataclass
class ThreeLayerFloorplanController:
    token_head: FloorplanTokenHead
    geometry_refiner: TokenVectorRefiner
    topology_head: TokenGraphResidualHead | None
    geometry_codetree: CodeTreeAnchorDecoder | None

    def room_and_geometry_transition(
        self,
        output: dict[str, torch.Tensor],
        token_ids: torch.Tensor,
        token_mask: torch.Tensor | None,
        token_stage_ids: torch.Tensor | None,
        token_input_ids: torch.Tensor | None,
        token_rollout_steps: int,
    ) -> tuple[dict[str, torch.Tensor], TypedFloorplanState]:
        refined = apply_token_refinement(
            self.token_head,
            self.geometry_refiner,
            output,
            token_ids,
            token_mask,
            token_stage_ids,
            token_input_ids,
            token_rollout_steps,
        )
        return refined, TypedFloorplanState.from_predictions(refined)

    def topology_transition(
        self,
        refined: dict[str, torch.Tensor],
        state: TypedFloorplanState,
        token_mask: torch.Tensor,
        token_stage_ids: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], TypedFloorplanState]:
        if self.topology_head is None:
            return refined, state
        refined = self.topology_head(
            refined,
            refined["token_refinement_hidden"],
            token_mask,
            token_stage_ids,
            refined.get("token_refiner_confidence"),
        )
        return refined, TypedFloorplanState.from_predictions(refined)

    def geometry_transition(
        self,
        refined: dict[str, torch.Tensor],
        state: TypedFloorplanState,
        token_mask: torch.Tensor,
        token_stage_ids: torch.Tensor,
        *,
        codetree_primary: bool,
        differentiable_alignment: bool = False,
    ) -> tuple[dict[str, torch.Tensor], TypedFloorplanState]:
        if self.geometry_codetree is None:
            return refined, state
        codetree = self.geometry_codetree(
            refined["token_refinement_hidden"],
            token_mask,
            token_stage_ids,
            predictions=refined,
        )
        base_state = state.clone()
        codetree_state = TypedFloorplanState.from_predictions(
            {
                **refined,
                **codetree,
                "room_polygons": codetree["codetree_polygons"],
            }
        ).align_rooms_to(
            base_state,
            differentiable=bool(differentiable_alignment),
        )
        refined.update(codetree)
        refined["codetree_polygons_aligned"] = codetree_state.room_polygons
        if codetree_primary and codetree_state.room_polygons is not None:
            state = codetree_state
            state.room_features = base_state.room_features
            state.room_instance_ids = base_state.room_instance_ids
        refined["typed_stage_layers"] = stage_layers_dict(state)
        return refined, state


__all__ = ["ThreeLayerFloorplanController"]

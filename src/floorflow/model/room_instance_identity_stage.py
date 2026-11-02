"""Instance-aware room identity stage with room-conditioned structure handoff."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.heads.room_adjacency import (
    RoomAdjacencyHead,
    RoomGraphConditioner,
    mask_contact_logits,
    owner_swap_metrics,
)
from floorflow.heads.room_mask_decoder import (
    RoomMaskDecoder,
    query_existence_from_class_logits,
)
from floorflow.heads.room_mask_fusion import (
    apply_class_count_prior,
    fused_room_mask_logits,
    query_gate_probability,
)
from floorflow.heads.room_owner_readout import (
    exclusive_query_owner_probability,
    room_owner_readout,
)
from floorflow.heads.room_query_set_head import (
    RoomQuerySetHead,
    query_count_selection_from_presence,
)
from floorflow.model.program_instance_router import ProgramInstanceRouter


def owner_aware_boundary(room_probability: torch.Tensor) -> torch.Tensor:
    """Combine spatial, second-owner, and margin uncertainty for structure."""

    spatial = F.max_pool2d(
        room_probability,
        3,
        stride=1,
        padding=1,
    ) - room_probability
    spatial = spatial.abs().amax(dim=1, keepdim=True)
    count = min(2, room_probability.shape[1])
    top_two = room_probability.topk(count, dim=1).values
    if count == 1:
        top_two = torch.cat([top_two, torch.zeros_like(top_two)], dim=1)
    owner_uncertainty = 1.0 - top_two[:, :1]
    second_owner = top_two[:, 1:2]
    owner_margin = 1.0 - (top_two[:, :1] - top_two[:, 1:2])
    return torch.stack(
        [spatial, owner_uncertainty, second_owner, owner_margin],
        dim=-1,
    ).amax(dim=-1)


def _sync_query_count_selection(
    prediction: dict[str, torch.Tensor],
    *,
    enabled: bool,
    training: bool,
) -> None:
    """Keep count-gated slots synchronized after logits change."""

    count_logits = prediction.get("room_count_logits")
    if count_logits is None:
        return
    selection = query_count_selection_from_presence(
        prediction["query_existence_logits"],
        count_logits,
    )
    prediction["query_count_candidate_mask"] = selection["selection"]
    prediction["query_count_soft_selection"] = selection["soft_selection"]
    prediction["query_count_ordered_indices"] = selection["ordered_indices"]
    if enabled:
        prediction["query_count_selection_mask"] = (
            selection["soft_selection"]
            if training
            else selection["selection"]
        )


class RoomInstanceIdentityStage(nn.Module):
    """Refine dense room identity with query masks and expose structure context."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        *,
        hidden_dim: int = 96,
        num_queries: int = 32,
        layers: int = 2,
        heads: int = 4,
        enable_program_instance_router: bool = False,
        predict_room_adjacency: bool = False,
        enable_query_set_head: bool = False,
        query_set_hidden: int = 64,
        query_set_pool: int = 48,
        query_set_layers: int = 2,
        query_set_routing_attention: bool = True,
        room_class_prior: tuple[float, ...] | None = None,
        query_count_gate: bool = True,
        owner_preserving_room_readout: bool | str = True,
        adaptive_query_refinement: bool = False,
        adaptive_query_residual_scale: float = 0.0,
        mask_aware_room_adjacency: bool = False,
        relational_room_adjacency: bool = False,
        relational_adjacency_hidden: int = 128,
        count_aware_room_graph: bool = False,
        count_aware_second_graph: bool = False,
        area_preserving_mask_residual: bool | str = False,
        full_resolution_mask_refinement: bool = False,
    ) -> None:
        super().__init__()
        self.decoder = RoomMaskDecoder(
            in_channels,
            num_room_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            layers=layers,
            heads=heads,
            feature_scales=(1, 2, 4),
            program_token_dim=in_channels,
            adaptive_query_refinement=adaptive_query_refinement,
            adaptive_query_residual_scale=adaptive_query_residual_scale,
            full_resolution_mask_refinement=(
                full_resolution_mask_refinement
            ),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.program_router = (
            ProgramInstanceRouter(hidden_dim, num_room_classes)
            if enable_program_instance_router
            else None
        )
        self.structure_adapter = nn.Sequential(
            nn.Conv2d(num_room_classes + 1, in_channels, 3, padding=1),
            nn.GroupNorm(max(1, in_channels // 16), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
        nn.init.normal_(
            self.structure_adapter[-1].weight,
            mean=0.0,
            std=1.0e-3,
        )
        nn.init.zeros_(self.structure_adapter[-1].bias)
        self.room_adjacency_head = (
            RoomAdjacencyHead(
                hidden_dim,
                mask_aware=mask_aware_room_adjacency,
                relational=relational_room_adjacency,
                relational_hidden=relational_adjacency_hidden,
            )
            if predict_room_adjacency
            else None
        )
        self.mask_aware_room_adjacency = bool(
            predict_room_adjacency and mask_aware_room_adjacency
        )
        self.count_aware_room_graph = bool(
            predict_room_adjacency
            and enable_query_set_head
            and count_aware_room_graph
        )
        self.count_aware_second_graph = bool(
            predict_room_adjacency
            and enable_query_set_head
            and count_aware_second_graph
        )
        self.room_graph_conditioner = (
            RoomGraphConditioner(
                hidden_dim,
                in_channels,
                num_room_classes,
                area_preserving_mask_residual=area_preserving_mask_residual,
            )
            if predict_room_adjacency
            else None
        )
        self.query_set_head = (
            RoomQuerySetHead(
                hidden_dim,
                in_channels,
                num_room_classes,
                num_queries,
                hidden_dim=query_set_hidden,
                pool_size=query_set_pool,
                attention_layers=query_set_layers,
                attention_heads=heads,
                routing_attention=query_set_routing_attention,
                class_count_prior=(
                    torch.tensor(room_class_prior)
                    if room_class_prior is not None
                    else None
                ),
            )
            if enable_query_set_head
            else None
        )
        self.query_count_gate = bool(query_count_gate)
        self.owner_readout_mode = (
            "relative"
            if owner_preserving_room_readout == "relative"
            else "absolute"
        )
        self.owner_preserving_room_readout = bool(
            owner_preserving_room_readout
        )
        self.class_count_prior_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        features: torch.Tensor,
        dense_room_logits: torch.Tensor,
        room_action_token: torch.Tensor | None = None,
        program: torch.Tensor | None = None,
        program_action_token: torch.Tensor | None = None,
        room_stage_candidate_logits: torch.Tensor | None = None,
        full_resolution_room_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prototype_logits = dense_room_logits
        stage_prior_scale = 0.5 * torch.tanh(self.residual_scale)
        if room_stage_candidate_logits is not None:
            if room_stage_candidate_logits.shape != dense_room_logits.shape:
                raise ValueError(
                    "room stage candidate and dense logits must share a shape"
                )
            prototype_logits = (
                dense_room_logits
                + stage_prior_scale
                * (room_stage_candidate_logits - dense_room_logits)
            )
        program_tokens = None
        if room_action_token is not None:
            routed_room_token = room_action_token
            if program_action_token is not None:
                if program_action_token.shape != room_action_token.shape:
                    raise ValueError(
                        "program action token must match the room action token"
                    )
                routed_room_token = routed_room_token + program_action_token
            program_tokens = routed_room_token[:, None]
        prediction = self.decoder(
            features,
            prototype_logits=prototype_logits,
            program_tokens=program_tokens,
            full_resolution_prototype_logits=(
                full_resolution_room_logits
            ),
        )
        if self.program_router is not None:
            if program is None:
                raise ValueError(
                    "program tensor is required by the instance router"
                )
            assignment = self.program_router(
                prediction["query_features"],
                program,
            )
            prediction["class_logits"] = (
                prediction["class_logits"]
                + torch.tanh(self.program_router.scale)
                * assignment["room_class_assignment_logits"]
            )
            prediction["program_room_assignment_logits"] = assignment[
                "assignment_logits"
            ]
            prediction["program_room_class_logits"] = assignment[
                "room_class_assignment_logits"
            ]
            prediction["program_room_scheduled_logits"] = assignment[
                "scheduled_assignment_logits"
            ]
            prediction["program_room_scheduled_class_logits"] = assignment[
                "scheduled_room_class_logits"
            ]
            prediction["program_room_assignment_scale"] = assignment[
                "assignment_scale"
            ]
        adjacency_logits = None
        if (
            self.room_adjacency_head is not None
            and self.room_graph_conditioner is not None
        ):
            prediction["query_existence_logits"] = (
                query_existence_from_class_logits(
                    prediction["class_logits"]
                )
            )
            graph_presence = prediction["query_existence_logits"].sigmoid()
            if (
                self.count_aware_room_graph
                and self.query_set_head is not None
            ):
                draft_set = self.query_set_head(
                    features,
                    prediction["query_features"],
                    prediction["mask_logits"],
                    prediction["query_existence_logits"],
                )
                draft_selection = (
                    draft_set["query_count_soft_selection"]
                    if self.training
                    else draft_set["query_count_candidate_mask"]
                )
                graph_presence = graph_presence * draft_selection.to(
                    graph_presence.dtype
                )
            prediction["room_graph_query_presence"] = graph_presence
            adjacency_logits = self.room_adjacency_head(
                prediction["query_features"],
                prediction["mask_logits"],
                query_presence=graph_presence,
            )
            pre_graph_mask_logits = prediction["mask_logits"]
            graph = self.room_graph_conditioner(
                prediction["query_features"],
                adjacency_logits,
                features,
                query_presence=graph_presence,
                mask_logits=pre_graph_mask_logits,
            )
            prediction["query_features"] = graph["query_features"]
            class_logits = prediction["class_logits"].clone()
            class_logits[:, :, : graph["class_residual"].shape[-1]] += graph[
                "class_residual"
            ]
            prediction["class_logits"] = class_logits
            prediction["room_graph_first_class_residual"] = graph[
                "class_residual"
            ]
            mask_residual = graph["mask_residual"]
            if mask_residual.shape[-2:] != prediction["mask_logits"].shape[-2:]:
                mask_residual = F.interpolate(
                    mask_residual,
                    size=prediction["mask_logits"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            prediction["mask_logits"] = (
                prediction["mask_logits"] + mask_residual
            )
            first_swap = owner_swap_metrics(
                pre_graph_mask_logits,
                prediction["mask_logits"],
                graph_presence,
            )
            prediction["room_graph_first_owner_swap_fraction"] = first_swap[
                "owner_swap_fraction"
            ]
            prediction["room_graph_first_owner_swap_support_fraction"] = (
                first_swap["owner_swap_support_fraction"]
            )
            prediction["room_graph_first_mask_residual"] = mask_residual
            prediction["query_existence_logits"] = (
                query_existence_from_class_logits(
                    prediction["class_logits"]
                )
            )
            prediction["room_graph_conditioning_scale"] = graph[
                "graph_conditioning_scale"
            ]
        query_set_residual = None
        if self.query_set_head is not None:
            pre_query_existence = query_existence_from_class_logits(
                prediction["class_logits"]
            )
            query_set = self.query_set_head(
                features,
                prediction["query_features"],
                prediction["mask_logits"],
                pre_query_existence,
            )
            query_set_residual = (
                query_set["query_existence_logits"]
                - pre_query_existence
            )
            prediction.update(query_set)
            if "room_class_count_logits" in query_set:
                prior_scale = torch.tanh(self.class_count_prior_scale)
                prediction["class_logits"] = apply_class_count_prior(
                    prediction["class_logits"],
                    query_set["room_class_count_logits"],
                    scale=prior_scale,
                )
                prediction["room_class_count_prior_scale"] = prior_scale
            class_existence = query_existence_from_class_logits(
                prediction["class_logits"]
            )
            prediction["query_existence_logits"] = (
                class_existence + query_set_residual
            )
            _sync_query_count_selection(
                prediction,
                enabled=self.query_count_gate,
                training=self.training,
            )
        final_presence = prediction["query_existence_logits"].sigmoid()
        selection = prediction.get("query_count_selection_mask")
        if selection is not None:
            final_presence = query_gate_probability(
                prediction["query_existence_logits"],
                selection,
            )
            prediction["room_query_selected_count"] = (
                selection.float().sum(dim=1)
            )
        prediction["room_graph_query_presence"] = final_presence
        final_adjacency = None
        if self.count_aware_second_graph:
            if (
                self.room_adjacency_head is None
                or self.room_graph_conditioner is None
            ):
                raise ValueError(
                    "second graph pass requires adjacency conditioning"
                )
            final_adjacency = self.room_adjacency_head(
                prediction["query_features"],
                prediction["mask_logits"],
                query_presence=final_presence,
            )
            pre_second_mask_logits = prediction["mask_logits"]
            graph = self.room_graph_conditioner(
                prediction["query_features"],
                final_adjacency,
                features,
                query_presence=final_presence,
                mask_logits=pre_second_mask_logits,
            )
            prediction["query_features"] = graph["query_features"]
            updated_class = prediction["class_logits"].clone()
            updated_class[
                :, :, : graph["class_residual"].shape[-1]
            ] += graph["class_residual"]
            prediction["class_logits"] = updated_class
            prediction["room_graph_second_class_residual"] = graph[
                "class_residual"
            ]
            mask_residual = graph["mask_residual"]
            if mask_residual.shape[-2:] != prediction["mask_logits"].shape[-2:]:
                mask_residual = F.interpolate(
                    mask_residual,
                    size=prediction["mask_logits"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            prediction["mask_logits"] = (
                prediction["mask_logits"] + mask_residual
            )
            second_swap = owner_swap_metrics(
                pre_second_mask_logits,
                prediction["mask_logits"],
                final_presence,
            )
            prediction["room_graph_second_owner_swap_fraction"] = (
                second_swap["owner_swap_fraction"]
            )
            prediction["room_graph_second_owner_swap_support_fraction"] = (
                second_swap["owner_swap_support_fraction"]
            )
            prediction["room_graph_second_mask_residual"] = mask_residual
            prediction["query_existence_logits"] = (
                query_existence_from_class_logits(
                    prediction["class_logits"]
                )
                + (
                    query_set_residual
                    if query_set_residual is not None
                    else 0.0
                )
            )
            _sync_query_count_selection(
                prediction,
                enabled=self.query_count_gate,
                training=self.training,
            )
            selection = prediction.get("query_count_selection_mask")
            selected = (
                selection
                if selection is not None
                else torch.ones(
                    final_presence.shape,
                    dtype=torch.bool,
                    device=final_presence.device,
                )
            )
            final_presence = query_gate_probability(
                prediction["query_existence_logits"],
                selected,
            )
            prediction["room_query_selected_count"] = (
                selected.float().sum(dim=1)
            )
            prediction["room_graph_query_presence"] = final_presence
            prediction["room_graph_second_pass_scale"] = graph[
                "graph_conditioning_scale"
            ]
        prediction["slot_owner_probability"] = (
            exclusive_query_owner_probability(prediction["mask_logits"])
        )
        fused = fused_room_mask_logits(
            prediction,
            dense_room_logits.shape[1],
            fusion_mode="sigmoid_existence",
        )
        if self.owner_preserving_room_readout:
            owner = room_owner_readout(
                prediction,
                dense_room_logits.shape[1],
                fallback_logits=fused,
                soft_foreground=self.training,
                confidence_mode=self.owner_readout_mode,
            )
            fused = owner["owner_preserving_logits"]
            prediction.update(
                {
                    "room_owner_map": owner["owner_map"],
                    "room_owner_foreground": owner["owner_foreground"],
                    "room_owner_boundary": owner["owner_boundary"],
                    "room_owner_soft_weight_mean": owner[
                        "owner_soft_weight"
                    ].float().mean(),
                    "room_owner_relative_confidence_mean": owner[
                        "owner_relative_confidence"
                    ].float().mean(),
                    "room_owner_foreground_fraction": owner[
                        "owner_foreground"
                    ].float().mean(),
                    "room_owner_mass_sum_mean": owner[
                        "owner_mass_sum"
                    ].float().mean(),
                }
            )
        if self.decoder.full_resolution_mask_scale is not None:
            prediction["room_full_resolution_mask_scale"] = torch.tanh(
                self.decoder.full_resolution_mask_scale
            )
        dense_fused = (
            F.interpolate(
                fused.float(),
                size=dense_room_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if fused.shape[-2:] != dense_room_logits.shape[-2:]
            else fused
        )
        prediction["room_owner_fused_logits"] = fused
        residual = dense_fused - dense_fused.mean(dim=1, keepdim=True)
        scaled_residual = torch.tanh(self.residual_scale) * residual
        prediction["room_query_residual_abs_mean"] = (
            scaled_residual.float().abs().mean()
        )
        refined = (
            dense_room_logits
            + scaled_residual
        )
        room_probability = refined.softmax(dim=1)
        boundary = owner_aware_boundary(room_probability)
        structure_context = torch.cat([room_probability, boundary], dim=1)
        structure_features = features + self.structure_adapter(
            structure_context
        )
        output = {
            **prediction,
            "room_instance_logits": refined,
            "room_instance_probability": room_probability,
            "room_stage_prior_scale": stage_prior_scale,
            "structure_features": structure_features,
        }
        if self.room_adjacency_head is not None:
            output["room_adjacency_logits"] = self.room_adjacency_head(
                prediction["query_features"],
                prediction["mask_logits"],
                query_presence=final_presence,
            )
        if self.mask_aware_room_adjacency:
            output["room_adjacency_mask_contact_logits"] = (
                mask_contact_logits(prediction["mask_logits"])
            )
        return output


__all__ = ["RoomInstanceIdentityStage", "owner_aware_boundary"]

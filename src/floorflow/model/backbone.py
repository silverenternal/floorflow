"""Conditional velocity model for boundary-to-floorplan flow matching."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.flow.coarse_to_face import CoarseToFaceFlowHead
from floorflow.heads.dynamic_output import DynamicMultiScaleOutputHead
from floorflow.heads.geometry_heads import GeometryHeatmapHeads
from floorflow.heads.polygon_loop import PolygonLoopHead
from floorflow.heads.room_groups import ROOM_GROUPS
from floorflow.heads.room_mask_decoder import RoomMaskDecoder
from floorflow.heads.room_partition import RoomPartitionHead
from floorflow.heads.room_program import RoomProgramHead
from floorflow.heads.semantic_head import SemanticEndpointHead
from floorflow.heads.spatial_velocity_head import SpatialVelocityHead
from floorflow.heads.structured_objects import StructuredObjectHeads
from floorflow.heads.topology import TopologyAuxHeads
from floorflow.model.anchor_fusion import SemanticAnchorFusion
from floorflow.model.backbone_aux import run_forward_with_aux
from floorflow.model.budget import resolve_output_kernels
from floorflow.model.cfm_backbone_mixin import CFMBackboneMixin
from floorflow.model.cfm_structure_adapter import build_cfm_structure_modules
from floorflow.model.family_stream_mixer import SemanticFamilyStreamMixer
from floorflow.model.flow_blocks import (
    ConditionEncoder,
    Downsample,
    GatedConditionCrossAttention,
    TimeEmbedding,
    TimestepBlock,
    Upsample,
)
from floorflow.model.flow_blocks import (
    group_count as _group_count,
)
from floorflow.model.hierarchical_pipeline import HierarchicalRoomPipeline
from floorflow.model.layout_stream_mixer import LayoutStreamMixer
from floorflow.model.layoutbridge_adapters import AnisotropicVelocityRefinement, DualStreamStructureInjection
from floorflow.model.semantic_attention import SemanticChannelCrossAttention, SemanticSlotFlowMixer
from floorflow.representation.program_conditioning import ProgramConditioner


class FloorplanFlowModel(CFMBackboneMixin, nn.Module):
    """U-Net velocity field conditioned on a floorplan exterior contour."""

    def __init__(
        self,
        target_channels: int = 12,
        base_channels: int = 48,
        levels: int = 3,
        condition_dropout: float = 0.05,
        semantic_channel_attention: bool = True,
        semantic_relation_bias: bool = True,
        slot_mixing: bool = True,
        num_slots: int = 16,
        dynamic_output_head: bool = True,
        output_kernels: tuple[int, ...] | None = None,
        input_resolution: tuple[int, int] = (256, 256),
        floor_conditioning: bool = True,
        num_floors: int = 4,
        topology_aux: bool = True,
        num_rooms: int = 1,
        semantic_endpoint_head: bool = True,
        channel_relations: tuple[tuple[float, ...], ...] | None = None,
        advanced_streams: bool = False,
        family_streams: bool = False,
        family_mita_attention: bool = False,
        family_attention_impl: str = "legacy",
        fourier_conditioning: bool = True,
        room_partition_head: bool = False,
        num_room_classes: int = 0,
        room_partition_semantic_prior: bool = False,
        room_class_ids: tuple[int, ...] | None = None,
        room_program_head: bool = False,
        room_group_head: bool = False,
        hierarchical_rooms: bool = False,
        hierarchical_hidden: int = 96,
        hierarchical_tokens: int = 8,
        hierarchical_coarse_resolution: int = 64,
        polygon_loop_head: bool = False,
        polygon_queries: int = 24,
        polygon_points: int = 24,
        polygon_hidden: int = 128,
        layoutbridge_refinement: bool = False,
        layoutbridge_hidden: int = 64,
        room_mask_head: bool = False,
        room_mask_queries: int = 32,
        room_mask_resolution: int = 64,
        room_mask_use_partition: bool = True,
        structured_object_head: bool = False,
        semantic_taxonomy: FloorplanTaxonomy | None = None,
        semantic_anchor_fusion: bool = False,
        geometry_heads: bool = False,
        geometry_hidden: int = 64,
        geometry_experts: int = 1,
        geometry_top_k: int = 1,
        coarse_to_face_cfm: bool = False,
        coarse_face_hidden: int = 64,
        coarse_face_max_faces: int = 32,
        program_conditioning: bool = False,
        program_hidden: int = 64,
        static_condition_encoder: bool = False,
        typed_family_streams: bool = False,
        multiscale_structure_injection: bool = False,
    ) -> None:
        super().__init__()
        if levels < 2: raise ValueError("levels must be at least 2")
        self.target_channels = int(target_channels)
        self.levels = int(levels)
        self.condition_dropout = float(condition_dropout)
        self.use_semantic_channel_attention = bool(semantic_channel_attention)
        self.use_semantic_relation_bias = bool(semantic_relation_bias)
        self.use_slot_mixing = bool(slot_mixing)
        self.use_dynamic_output_head = bool(dynamic_output_head)
        self.input_resolution, self.output_kernels = resolve_output_kernels(
            output_kernels, input_resolution, base_channels, self.target_channels
        )
        self.use_floor_conditioning = bool(floor_conditioning)
        self.num_floors = int(num_floors)
        self.use_topology_aux = bool(topology_aux)
        self.num_rooms = int(num_rooms)
        self.use_semantic_endpoint_head = bool(semantic_endpoint_head)
        self.use_advanced_streams = bool(advanced_streams)
        self.use_family_streams = bool(family_streams)
        self.use_family_mita_attention = bool(family_mita_attention)
        self.family_attention_impl = str(family_attention_impl)
        self.use_fourier_conditioning = bool(fourier_conditioning)
        self.use_room_partition_head = bool(room_partition_head)
        self.num_room_classes = int(num_room_classes)
        self.use_room_partition_semantic_prior = bool(room_partition_semantic_prior)
        self.use_room_program_head = bool(room_program_head)
        self.use_room_group_head = bool(room_group_head)
        self.use_hierarchical_rooms = bool(hierarchical_rooms)
        self.use_polygon_loop_head = bool(polygon_loop_head)
        self.semantic_taxonomy = semantic_taxonomy
        if self.use_room_group_head and self.semantic_taxonomy is None: raise ValueError("room_group_head requires semantic_taxonomy")
        if self.use_room_partition_semantic_prior and not self.use_semantic_endpoint_head: raise ValueError("room-partition semantic prior requires the semantic endpoint head")
        if self.use_hierarchical_rooms and not self.use_room_partition_head:
            raise ValueError("hierarchical rooms require the room-partition head")
        if self.use_room_partition_semantic_prior:
            if room_class_ids is None or len(room_class_ids) != self.num_room_classes:
                raise ValueError("room_class_ids must match num_room_classes for the semantic prior")
            self.register_buffer(
                "room_class_ids",
                torch.tensor(room_class_ids, dtype=torch.long),
                persistent=False,
            )
        self.use_layoutbridge_refinement = bool(layoutbridge_refinement)
        self.use_room_mask_head = bool(room_mask_head)
        self.room_mask_use_partition = bool(room_mask_use_partition)
        self.use_structured_object_head = bool(structured_object_head)
        self.use_semantic_anchor_fusion = bool(semantic_anchor_fusion)
        self.use_geometry_heads = bool(geometry_heads)
        self.use_coarse_to_face_cfm = bool(coarse_to_face_cfm)
        self.use_program_conditioning = bool(program_conditioning)
        self.static_condition_encoder = bool(static_condition_encoder)
        self.typed_family_streams = bool(typed_family_streams)
        self.multiscale_structure_injection = bool(
            multiscale_structure_injection
        )
        time_channels = int(base_channels) * 4
        self.time_embedding = TimeEmbedding(time_channels)
        self.floor_embedding = nn.Embedding(self.num_floors + 1, time_channels) if self.use_floor_conditioning else nn.Identity()
        self.condition_channels = 10 if self.use_fourier_conditioning else 6
        self.condition_encoder = ConditionEncoder(
            self.condition_channels,
            base_channels,
            levels,
            use_time=not self.static_condition_encoder,
        )

        channels = [base_channels * (2**level) for level in range(levels)]
        (
            self.family_stream_evidence,
            self.structure_token_encoder,
            self.structure_injector,
            self.velocity_family_streams,
        ) = build_cfm_structure_modules(
            target_channels,
            channels[-1],
            tuple(channels),
            semantic_taxonomy,
            typed_family_streams=self.typed_family_streams,
            multiscale_structure_injection=self.multiscale_structure_injection,
        )
        self.input_projection = nn.Conv2d(target_channels, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList(
            [TimestepBlock(channels[level], channels[level], time_channels) for level in range(levels)]
        )
        self.encoder_down = nn.ModuleList(
            [Downsample(channels[level], channels[level + 1]) for level in range(levels - 1)]
        )
        self.middle = TimestepBlock(channels[-1], channels[-1], time_channels)
        self.condition_attention = GatedConditionCrossAttention(channels[-1])
        self.slot_mixer = (
            SemanticSlotFlowMixer(channels[-1], num_slots=num_slots) if self.use_slot_mixing else nn.Identity()
        )
        self.semantic_channel_attention = (
            SemanticChannelCrossAttention(
                channels[-1],
                target_channels,
                use_relation_bias=self.use_semantic_relation_bias,
                relation_prior=channel_relations,
            )
            if self.use_semantic_channel_attention
            else nn.Identity()
        )
        if self.use_layoutbridge_refinement:
            self.dual_stream_injection = DualStreamStructureInjection(channels[-1], channels[-1])
            self.anisotropic_velocity = AnisotropicVelocityRefinement(
                target_channels,
                channels[0],
                layoutbridge_hidden,
            )
        else:
            self.dual_stream_injection = nn.Identity()
            self.anisotropic_velocity = nn.Identity()
        self.decoder_up = nn.ModuleList([Upsample(channels[level]) for level in range(levels - 1, 0, -1)])
        self.decoder = nn.ModuleList(
            [
                TimestepBlock(channels[level] + channels[level - 1], channels[level - 1], time_channels)
                for level in range(levels - 1, 0, -1)
            ]
        )
        self.output_norm = nn.GroupNorm(_group_count(channels[0]), channels[0])
        self.output = nn.Conv2d(channels[0], target_channels, 3, padding=1)
        self.topology_aux = (
            TopologyAuxHeads(channels[0], self.num_rooms) if self.use_topology_aux else nn.Identity()
        )
        self.semantic_endpoint = (
            SemanticEndpointHead(channels[0], target_channels)
            if self.use_semantic_endpoint_head
            else nn.Identity()
        )
        self.dynamic_output_head = (
            DynamicMultiScaleOutputHead(channels[0], target_channels, self.output_kernels)
            if self.use_dynamic_output_head
            else nn.Identity()
        )
        self.layout_stream_mixer = (
            LayoutStreamMixer(
                channels[-1],
                ("room", "wall", "opening", "furniture", "condition", "direct"),
            )
            if self.use_advanced_streams
            else nn.Identity()
        )
        self.family_stream_mixer = (
            SemanticFamilyStreamMixer(
                channels[-1],
                attention_type="mita" if self.use_family_mita_attention else "mha",
                attention_impl=self.family_attention_impl,
            )
            if self.use_family_streams
            else nn.Identity()
        )
        self.spatial_velocity = (
            SpatialVelocityHead(
                channels[0],
                ("room", "wall", "opening", "furniture", "condition"),
                output_dim=target_channels,
            )
            if self.use_advanced_streams
            else nn.Identity()
        )
        self.room_partition = (
            RoomPartitionHead(
                channels[0],
                self.num_room_classes,
                prior_channels=(
                    self.num_room_classes
                    if self.use_room_partition_semantic_prior
                    else 0
                ),
            )
            if self.use_room_partition_head
            else nn.Identity()
        )
        self.room_program = (
            RoomProgramHead(channels[0], self.num_room_classes)
            if self.use_room_program_head and self.num_room_classes > 0
            else nn.Identity()
        )
        self.room_program_scale = nn.Parameter(torch.ones(1))
        self.program_conditioner = (
            ProgramConditioner(self.num_room_classes, hidden_dim=program_hidden)
            if self.use_program_conditioning
            else nn.Identity()
        )
        self.program_scale = (
            nn.Parameter(torch.zeros(1))
            if self.use_program_conditioning
            else None
        )
        self.room_group = (
            RoomPartitionHead(channels[0], len(ROOM_GROUPS))
            if self.use_room_group_head
            else nn.Identity()
        )
        self.room_group_scale = nn.Parameter(torch.ones(1))
        self.hierarchical_rooms = (
            HierarchicalRoomPipeline(
                channels[0],
                self.num_room_classes,
                hidden_dim=hierarchical_hidden,
                tokens=hierarchical_tokens,
                coarse_resolution=hierarchical_coarse_resolution,
                prior_channels=(
                    self.num_room_classes
                    if self.use_room_partition_semantic_prior
                    else 0
                ),
            )
            if self.use_hierarchical_rooms and self.num_room_classes > 0
            else nn.Identity()
        )
        self.hierarchical_scale = nn.Parameter(torch.ones(1))
        self.coarse_room_scale = nn.Parameter(torch.zeros(1))
        self.hierarchical_program_scale = nn.Parameter(torch.zeros(1))
        self.polygon_loop = (
            PolygonLoopHead(
                channels[0],
                self.num_room_classes,
                hidden_dim=polygon_hidden,
                num_queries=polygon_queries,
                points=polygon_points,
                class_token_dim=hierarchical_hidden,
            )
            if self.use_polygon_loop_head and self.num_room_classes > 0
            else nn.Identity()
        )
        if self.use_room_partition_head and self.use_room_partition_semantic_prior:
            # The semantic prior is a trainable residual. Identity initialization
            # makes a zero-initialized partition head reproduce the semantic room
            # logits exactly, preserving the condition-only checkpoint.
            with torch.no_grad():
                self.room_partition.prior_logits.weight.zero_()
                for index in range(min(self.num_room_classes, self.room_partition.prior_logits.out_channels)):
                    self.room_partition.prior_logits.weight[index, index, 0, 0] = 1.0
        self.room_mask_decoder = (
            RoomMaskDecoder(
                channels[0]
                + (
                    self.num_room_classes
                    if self.use_room_partition_head and self.room_mask_use_partition
                    else 0
                ),
                self.num_room_classes,
                hidden_dim=128,
                num_queries=room_mask_queries,
            )
            if self.use_room_mask_head
            else nn.Identity()
        )
        self.room_mask_resolution = int(room_mask_resolution)
        self.semantic_anchor_fusion = (
            SemanticAnchorFusion(target_channels) if self.use_semantic_anchor_fusion else nn.Identity()
        )
        if self.use_structured_object_head:
            if semantic_taxonomy is None:
                raise ValueError("semantic_taxonomy is required for structured object heads")
            self.structured_objects = StructuredObjectHeads(channels[0], semantic_taxonomy)
        else:
            self.structured_objects = nn.Identity()
        self.geometry_heads = (
            GeometryHeatmapHeads(
                channels[0],
                hidden_dim=geometry_hidden,
                experts=geometry_experts,
                top_k=geometry_top_k,
            )
            if self.use_geometry_heads
            else nn.Identity()
        )
        self.coarse_to_face_cfm = (
            CoarseToFaceFlowHead(
                channels[0], target_channels, self.num_room_classes,
                hidden_dim=coarse_face_hidden, max_faces=coarse_face_max_faces,
                token_dim=hierarchical_hidden if self.use_hierarchical_rooms else 0,
            )
            if self.use_coarse_to_face_cfm
            else nn.Identity()
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        footprint: torch.Tensor,
        floor_ids: torch.Tensor | None = None,
        condition_features: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        output_features = self._encode_features(
            x_t,
            t,
            footprint,
            floor_ids=floor_ids,
            condition_features=condition_features,
        )
        return self._velocity_from_features(output_features, t)

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        footprint: torch.Tensor,
        floor_ids: torch.Tensor | None = None,
        detach_object_features: bool = False,
        layout_prior_logits: torch.Tensor | None = None,
        face_ids: torch.Tensor | None = None,
        room_program: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return run_forward_with_aux(
            self,
            x_t,
            t,
            footprint,
            floor_ids=floor_ids,
            detach_object_features=detach_object_features,
            layout_prior_logits=layout_prior_logits,
            face_ids=face_ids,
            room_program=room_program,
        )

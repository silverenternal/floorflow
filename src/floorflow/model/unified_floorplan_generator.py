"""Boundary-conditioned unified room and CAD-scaffold generator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.boundary_edge_graph import BoundaryEdgeGraphEncoder
from floorflow.geometry.boundary_geometry import BoundaryGeometryEncoder
from floorflow.geometry.entrance_condition import EntranceGeometryEncoder
from floorflow.geometry.partition_edge_evidence import partition_edge_evidence
from floorflow.heads.codetree_anchor_decoder import CodeTreeAnchorDecoder
from floorflow.heads.compact_boundary_encoder import CompactBoundaryEncoder
from floorflow.heads.floorplan_token_head import FloorplanTokenHead
from floorflow.heads.query_aware_pool import QueryAwarePool
from floorflow.heads.room_instance_token_head import RoomInstanceTokenHead
from floorflow.heads.room_mask_decoder import RoomMaskDecoder
from floorflow.heads.room_program_set import RoomProgramSetHead
from floorflow.heads.room_query_utils import mask_centers
from floorflow.heads.room_segmenter import ContourRoomSegmenter
from floorflow.heads.token_graph_residual_head import TokenGraphResidualHead
from floorflow.heads.token_vector_refiner import TokenVectorRefiner
from floorflow.heads.typed_token_memory import TypedTokenMemory
from floorflow.heads.vector_primitive_head import VectorPrimitiveHead
from floorflow.inference.generated_floorplan_readout import (
    decode_generated_floorplan,
)
from floorflow.inference.rplan_query_quality_arbitration import (
    make_rplan_query_quality_head,
    rplan_query_quality_features,
    score_rplan_query_quality,
)
from floorflow.model.cad_action_graph import STAGES, CadActionGraph
from floorflow.model.hierarchical_pipeline import RoomQueryGraphRefiner
from floorflow.model.official_mita_route import OfficialMiTARouteAttention
from floorflow.model.three_layer_controller import ThreeLayerFloorplanController
from floorflow.representation.typed_floorplan_state import TypedFloorplanState
from floorflow.representation.typed_state_transition import (
    transition_typed_state_with_tokens,
)


@dataclass(frozen=True)
class UnifiedGeneratorConfig:
    hidden_dim: int = 96
    decoder_channels: int = 96
    identity_queries: int = 12
    identity_layers: int = 1
    vector_layers: int = 1
    polygon_points: int = 24
    max_walls: int = 128
    max_openings: int = 8
    attention_heads: int = 4
    route_top_k: int = 16
    max_boundary_points: int = 64
    pretrained_backbone: bool = False
    use_mask_anchors: bool = True
    share_identity_heads: bool = True
    enable_query_quality: bool = False
    enable_point_validity: bool = False
    use_boundary_edge_graph: bool = False
    encoder_architecture: str = "resnet18"
    compact_encoder_channels: tuple[int, ...] = (64, 128, 256, 384)
    compact_encoder_blocks: tuple[int, ...] = (2, 4, 6, 3)
    enable_program_set: bool = False
    enable_query_graph_refiner: bool = False
    enable_entrance_condition: bool = False
    enable_partition_edge_condition: bool = False
    enable_partition_consistent_queries: bool = False
    enable_partition_attention_masking: bool = True
    enable_ordered_stage_context: bool = False
    ordered_stage_context_scale: float = 0.25
    enable_token_head: bool = False
    token_vocab_size: int = 0
    token_max_length: int = 512
    token_head_layers: int = 2
    token_refine_scale: float = 0.02
    token_refiner_min_confidence: float = 0.05
    enable_token_graph_head: bool = True
    token_graph_residual_scale: float = 0.5
    token_graph_max_junctions: int = 32
    token_graph_degree_classes: int = 9
    enable_codetree_head: bool = True
    codetree_primary: bool = True
    topology_faces_primary: bool = False


class UnifiedFloorplanGenerator(nn.Module):
    """Generate room identity, polygons, and a CAD scaffold from one boundary."""

    def __init__(
        self,
        taxonomy: FloorplanTaxonomy,
        config: UnifiedGeneratorConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or UnifiedGeneratorConfig()
        self.taxonomy = taxonomy
        self.config = config
        room_classes = len(taxonomy.room_ids)
        self.boundary_geometry = BoundaryGeometryEncoder(
            config.hidden_dim,
            max_points=config.max_boundary_points,
        )
        self.boundary_edges = (
            BoundaryEdgeGraphEncoder(config.hidden_dim)
            if config.use_boundary_edge_graph
            else None
        )
        self.spatial_encoder = (
            CompactBoundaryEncoder(
                room_classes,
                decoder_channels=config.decoder_channels,
                channels=tuple(config.compact_encoder_channels),
                blocks=tuple(config.compact_encoder_blocks),
            )
            if config.encoder_architecture == "compact"
            else ContourRoomSegmenter(
                room_classes,
                pretrained=config.pretrained_backbone,
                decoder_channels=config.decoder_channels,
            )
        )
        self.identity_decoder = RoomMaskDecoder(
            config.decoder_channels,
            room_classes,
            hidden_dim=config.hidden_dim,
            num_queries=config.identity_queries,
            layers=config.identity_layers,
            heads=config.attention_heads,
            feature_scales=(2, 4),
            program_token_dim=(
                config.hidden_dim if config.enable_program_set else 0
            ),
            partition_consistent=config.enable_partition_consistent_queries,
            partition_attention_masking=(
                config.enable_partition_attention_masking
            ),
        )
        self.program_set = (
            RoomProgramSetHead(
                config.decoder_channels,
                room_classes,
                hidden_dim=config.hidden_dim,
                tokens=8,
                layers=1,
                heads=config.attention_heads,
                num_queries=config.identity_queries,
            )
            if config.enable_program_set
            else None
        )
        self.identity_router = OfficialMiTARouteAttention(
            config.hidden_dim,
            num_heads=config.attention_heads,
            router_count=config.max_boundary_points,
            top_k=config.route_top_k,
        )
        self.vector_decoder = VectorPrimitiveHead(
            config.decoder_channels,
            room_classes,
            opening_classes=1,
            hidden_dim=config.hidden_dim,
            max_rooms=config.identity_queries,
            max_walls=config.max_walls,
            max_openings=config.max_openings,
            polygon_points=config.polygon_points,
            decoder_layers=config.vector_layers,
            attention_heads=config.attention_heads,
            stage_context=True,
            ordered_stage_context=config.enable_ordered_stage_context,
            stage_context_scale=config.ordered_stage_context_scale,
            host_conditioned_openings=True,
            point_validity=config.enable_point_validity,
            partition_edge_condition=config.enable_partition_edge_condition,
        )
        self.action_graph = CadActionGraph(
            {name: config.hidden_dim for name in STAGES},
            hidden_dim=config.hidden_dim,
            heads=config.attention_heads,
        )
        self.room_count_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.identity_queries + 1),
        )
        self.query_quality = (
            make_rplan_query_quality_head()
            if config.enable_query_quality
            else None
        )
        self.query_graph_refiner = (
            RoomQueryGraphRefiner(config.hidden_dim)
            if config.enable_query_graph_refiner
            else None
        )
        self.room_instance_head = (
            RoomInstanceTokenHead(config.hidden_dim, room_classes)
            if config.enable_token_head
            else None
        )
        self.token_head = (
            FloorplanTokenHead(
                config.hidden_dim,
                int(config.token_vocab_size),
                int(config.token_max_length),
                hidden_dim=config.hidden_dim,
                heads=config.attention_heads,
                layers=config.token_head_layers,
            )
            if config.enable_token_head
            else None
        )
        self.query_pool = (
            QueryAwarePool(config.hidden_dim)
            if config.enable_token_head
            else None
        )
        self.token_memory = (
            TypedTokenMemory(
                config.hidden_dim,
                ("envelope", "rooms", "walls", "openings"),
            )
            if config.enable_token_head
            else None
        )
        self.token_refiner = (
            TokenVectorRefiner(
                config.hidden_dim,
                max_walls=config.max_walls,
                max_openings=config.max_openings,
                polygon_points=config.polygon_points,
                heads=config.attention_heads,
                residual_scale=config.token_refine_scale,
                min_confidence=config.token_refiner_min_confidence,
            )
            if config.enable_token_head
            else None
        )
        self.token_graph_head = (
            TokenGraphResidualHead(
                config.hidden_dim,
                room_classes,
                max_rooms=config.identity_queries,
                max_walls=config.max_walls,
                max_openings=config.max_openings,
                max_junctions=config.token_graph_max_junctions,
                degree_classes=config.token_graph_degree_classes,
                residual_scale=config.token_graph_residual_scale,
                min_confidence=config.token_refiner_min_confidence,
            )
            if config.enable_token_head and config.enable_token_graph_head
            else None
        )
        self.codetree_head = (
            CodeTreeAnchorDecoder(
                config.hidden_dim,
                max(1, room_classes - 1),
                max_rooms=config.identity_queries,
                polygon_points=config.polygon_points,
                heads=config.attention_heads,
            )
            if config.enable_token_head and config.enable_codetree_head
            else None
        )
        self.stage_controller = (
            ThreeLayerFloorplanController(
                self.token_head,
                self.token_refiner,
                self.token_graph_head,
                self.codetree_head,
            )
            if self.token_head is not None and self.token_refiner is not None
            else None
        )
        self.entrance_geometry = None
        self.token_graph_catalogue = None
        if config.enable_entrance_condition:
            with torch.random.fork_rng(devices=[]):
                self.entrance_geometry = EntranceGeometryEncoder(
                    config.hidden_dim
                )

    def forward(
        self,
        boundary_mask: torch.Tensor,
        boundary_valid: torch.Tensor,
        boundary_points: torch.Tensor | None = None,
        entrance_points: torch.Tensor | None = None,
        entrance_valid: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        token_stage_ids: torch.Tensor | None = None,
        token_input_ids: torch.Tensor | None = None,
        token_rollout_steps: int = 0,
    ) -> dict[str, torch.Tensor]:
        if boundary_mask.ndim == 3:
            boundary_mask = boundary_mask[:, None]
        if boundary_mask.ndim != 4 or boundary_mask.shape[1] != 1:
            raise ValueError("boundary_mask must be [B,1,H,W]")
        if boundary_valid.ndim != 2:
            raise ValueError("boundary_valid must be [B,P]")
        if self.boundary_edges is None:
            geometry = self.boundary_geometry(boundary_mask)
            boundary_tokens = geometry["tokens"]
        else:
            if boundary_points is None:
                raise ValueError(
                    "boundary_points are required for boundary-edge encoding"
                )
            geometry = self.boundary_edges(
                boundary_points,
                boundary_valid,
                entrance_points,
                entrance_valid,
            )
            boundary_tokens = geometry["tokens"]
            boundary_valid = geometry["valid"]
        entrance = None
        if self.entrance_geometry is not None:
            if entrance_points is None or entrance_valid is None:
                raise ValueError(
                    "entrance points and validity are required when the "
                    "entrance condition is enabled"
                )
            entrance = self.entrance_geometry(entrance_points, entrance_valid)
            boundary_tokens = torch.cat(
                [boundary_tokens, entrance["tokens"]],
                dim=1,
            )
            boundary_valid = torch.cat(
                [boundary_valid, entrance["valid"][:, None]],
                dim=1,
            )
        features = self.spatial_encoder.encode_control(boundary_mask)
        program = (
            self.program_set(features)
            if self.program_set is not None
            else None
        )
        identity = self.identity_decoder(
            features,
            room_tokens=boundary_tokens,
            room_token_valid=boundary_valid,
            program_tokens=(
                program["class_tokens"] if program is not None else None
            ),
        )
        room_anchors = (
            mask_centers(identity["mask_logits"])
            if self.config.use_mask_anchors
            else None
        )
        valid_float = boundary_valid[..., None].to(boundary_tokens.dtype)
        boundary_global = (boundary_tokens * valid_float).sum(dim=1) / (
            valid_float.sum(dim=1).clamp_min(1.0)
        )
        identity_class_logits = identity["class_logits"]
        identity_presence = (
            torch.logsumexp(identity_class_logits[..., :-1], dim=-1)
            - identity_class_logits[..., -1]
        )
        room_global = (
            self.query_pool(
                identity["query_features"],
                identity_presence,
            )
            if self.query_pool is not None
            else identity["query_features"].mean(dim=1)
        )
        feature_global = features.mean(dim=(2, 3))
        action = self.action_graph(
            {
                "envelope": boundary_global,
                "program": boundary_global,
                "rooms": room_global,
                "walls": feature_global,
                "junctions": feature_global,
                "openings": boundary_global,
                "furniture": torch.zeros_like(feature_global),
                "validation": feature_global,
            }
        )
        stage_tokens = torch.stack(
            [action.tokens[name] for name in STAGES],
            dim=1,
        )
        routed_rooms, routing = self.identity_router(
            identity["query_features"],
            boundary_tokens,
            key_value_mask=boundary_valid,
        )
        routed_rooms = routed_rooms + action.tokens["rooms"][:, None]
        query_graph = None
        if self.query_graph_refiner is not None:
            query_graph = self.query_graph_refiner(
                routed_rooms,
                identity["mask_logits"],
                boundary_mask,
            )
            routed_rooms = query_graph["queries"]
            refined_class_queries = self.identity_decoder.class_fusion(
                routed_rooms,
                program["class_tokens"] if program is not None else None,
            )
            identity_class_logits = self.identity_decoder.class_head(
                self.identity_decoder.decoder_norm(refined_class_queries)
            )
        if self.room_instance_head is not None:
            instance_presence = (
                torch.logsumexp(identity_class_logits[..., :-1], dim=-1)
                - identity_class_logits[..., -1]
            )
            routed_rooms = self.room_instance_head(
                routed_rooms,
                identity_class_logits,
                instance_presence.sigmoid() >= 0.1,
            )
        graph_tokens = torch.cat([boundary_tokens, stage_tokens], dim=1)
        graph_mask = torch.cat(
            [
                boundary_valid,
                torch.ones(
                    boundary_valid.shape[0],
                    len(STAGES),
                    dtype=torch.bool,
                    device=boundary_valid.device,
                ),
            ],
            dim=1,
        )
        edge_condition = (
            partition_edge_evidence(identity["mask_logits"])
            if self.config.enable_partition_edge_condition
            else None
        )
        vectors = self.vector_decoder(
            features,
            graph_tokens=graph_tokens,
            graph_mask=graph_mask,
            room_query_prior=routed_rooms,
            room_query_anchors=room_anchors,
            partition_edge=edge_condition,
            stage_tokens=(
                action.tokens
                if self.config.enable_ordered_stage_context
                else None
            ),
        )
        if self.config.share_identity_heads:
            vectors["room_class"] = identity_class_logits[..., :-1]
            identity_presence = (
                torch.logsumexp(identity_class_logits[..., :-1], dim=-1)
                - identity_class_logits[..., -1]
            )
            vectors["room_exists"] = vectors["room_exists"] + identity_presence
        identity_presence = (
            torch.logsumexp(identity_class_logits[..., :-1], dim=-1)
            - identity_class_logits[..., -1]
        )
        if self.query_pool is not None:
            room_global = self.query_pool(routed_rooms, identity_presence)
        count_logits = self.room_count_head(
            torch.cat([boundary_global, room_global], dim=-1)
        )
        output = {
            **vectors,
            "room_count_logits": count_logits,
            "room_instance_class_logits": identity_class_logits,
            "room_instance_mask_logits": identity["mask_logits"],
            "room_instance_aux_mask_logits": identity["aux_mask_logits"],
            "room_instance_aux_class_logits": identity["aux_class_logits"],
            "room_identity_tokens": routed_rooms,
            "boundary_tokens": boundary_tokens,
            "boundary_valid": boundary_valid,
            "routing": routing,
            "action_tokens": action.tokens,
            "action_gates": action.gates,
            "support_mask": boundary_mask,
        }
        if self.token_head is not None:
            token_context, token_context_valid = self.token_memory(
                boundary_tokens,
                boundary_valid,
                routed_rooms,
                identity_presence.sigmoid() >= 0.1,
                stage_tokens,
                vectors["vector_stage_tokens"],
            )
            output["token_context_tokens"] = token_context
            output["token_context_valid"] = token_context_valid
            if self.token_graph_catalogue is not None:
                output["token_graph_catalogue"] = self.token_graph_catalogue.as_dict()
            if token_ids is not None:
                refined_output = self.refine_with_token_ids(
                    output,
                    token_ids,
                    token_mask,
                    token_stage_ids,
                    token_input_ids,
                    token_rollout_steps,
                )
                output.update(refined_output)
                output["typed_floorplan_state"] = (
                    TypedFloorplanState.from_predictions(output).to_predictions()
                )
        if "room_partition_logits" in identity:
            output["room_instance_partition_logits"] = identity[
                "room_partition_logits"
            ]
            output["room_instance_partition_probabilities"] = identity[
                "room_partition_probabilities"
            ]
        if query_graph is not None:
            output["room_query_graph_adjacency"] = query_graph["adjacency"]
            output["room_query_graph_descriptors"] = query_graph[
                "descriptors"
            ]
        if entrance is not None:
            output["entrance_condition_token"] = entrance["tokens"][:, 0]
        if edge_condition is not None:
            output["partition_edge_evidence"] = edge_condition
        if program is not None:
            output["room_program_class_logits"] = program["class_logits"]
            output["room_program_cardinality_logits"] = program[
                "cardinality_logits"
            ]
            output["room_program_presence_logits"] = program["presence_logits"]
            output["room_program_tokens"] = program["class_tokens"]
        if self.query_quality is not None:
            quality_features = rplan_query_quality_features(output)
            output["query_quality_features"] = quality_features
            output["query_quality_logits"] = score_rplan_query_quality(
                self.query_quality,
                quality_features,
            )
        return output

    def refine_with_token_ids(
        self,
        output: dict[str, torch.Tensor],
        token_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        token_stage_ids: torch.Tensor | None = None,
        token_input_ids: torch.Tensor | None = None,
        token_rollout_steps: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Refine cached base outputs without rerunning the encoder or vector decoder."""

        if self.token_head is None or self.token_refiner is None:
            raise ValueError("token refinement is not enabled")
        refined, _state = transition_typed_state_with_tokens(
            self.stage_controller,
            output,
            token_ids,
            token_mask,
            token_stage_ids,
            token_input_ids,
            token_rollout_steps,
            codetree_primary=self.config.codetree_primary,
            topology_faces_primary=self.config.topology_faces_primary,
            allow_hard_primary=not self.training,
            differentiable_alignment=self.training,
        )
        return refined
__all__ = [
    "UnifiedFloorplanGenerator",
    "UnifiedGeneratorConfig",
    "decode_generated_floorplan",
]

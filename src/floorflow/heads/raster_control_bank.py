"""Reusable raster-conditioned room-control models."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3,
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.full_resolution_class_sampler import (
    FullResolutionClassSampler,
)
from floorflow.heads.mask_pooled_class_fusion import MaskPooledClassFusion
from floorflow.heads.owner_semantic_reassignment import (
    OwnerSemanticReassignment,
)
from floorflow.heads.predicted_room_identity import PredictedRoomIdentityHead
from floorflow.heads.raster_control_adapters import (
    RasterDenseQueryRoomBlend,
    RasterInstanceClassRoutedAdapter,
    RasterInstanceRoomAdapter,
    RasterInstanceSoftRoutedAdapter,
)
from floorflow.heads.room_boundary_owner_refiner import (
    RoomBoundaryOwnerRefiner,
)
from floorflow.heads.room_mask_decoder import (
    RoomMaskDecoder,
    query_existence_from_class_logits,
)
from floorflow.heads.room_mask_fusion import (
    apply_program_class_prior,
    fused_room_mask_logits,
)
from floorflow.heads.room_model_contract import (
    raster_instance_module_contract,
)
from floorflow.heads.room_owner_readout import room_owner_readout
from floorflow.heads.room_program_set import RoomProgramSetHead
from floorflow.heads.room_query_set_head import RoomQuerySetHead
from floorflow.heads.room_segmenter import ContourRoomSegmenter, ConvNormAct
from floorflow.heads.room_slot_state import RoomSlotStateHead
from floorflow.heads.structured_objects import StructuredObjectHeads


class DeepLabRasterRoomControl(nn.Module):
    """DeepLabV3 room segmentation from a native or rendered floorplan image."""

    def __init__(
        self,
        num_room_classes: int,
        *,
        pretrained: bool = True,
        backbone_weights: str | Path | None = None,
    ) -> None:
        super().__init__()
        model: DeepLabV3 = deeplabv3_resnet50(
            weights=(
                DeepLabV3_ResNet50_Weights.DEFAULT
                if pretrained and backbone_weights is None
                else None
            ),
            weights_backbone=None,
        )
        if backbone_weights is not None and Path(backbone_weights).is_file():
            state = torch.load(
                backbone_weights,
                map_location="cpu",
                weights_only=True,
            )
            model.backbone.load_state_dict(state, strict=False)
        model.classifier[-1] = nn.Conv2d(
            256,
            int(num_room_classes),
            1,
        )
        self.model = model

    def forward(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if control_image is None:
            if footprint.ndim == 3:
                footprint = footprint[:, None]
            value = footprint.float()
            if value.shape[1] == 1:
                value = value.expand(-1, 3, -1, -1)
        else:
            value = control_image.float()
        logits = self.model(value)["out"]
        return F.interpolate(
            logits,
            size=footprint.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class RasterControlBank(nn.Module):
    """Joint room and structural control heads over one raster encoder."""

    def __init__(
        self,
        taxonomy: FloorplanTaxonomy,
        *,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        self.room = ContourRoomSegmenter(
            len(taxonomy.room_ids),
            pretrained=False,
            decoder_channels=int(decoder_channels),
        )
        self.object_heads = StructuredObjectHeads(
            int(decoder_channels),
            taxonomy,
        )

    def forward(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.room.encode_control(
            footprint,
            control_image=control_image,
        )
        return {
            "room_logits": self.room.final(hidden),
            **self.object_heads(hidden),
        }


class QueryMaskRefiner(nn.Module):
    """Refine query masks at the encoder's full spatial resolution."""

    def __init__(
        self,
        feature_channels: int,
        num_queries: int,
        *,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        self.feature = nn.Conv2d(int(feature_channels), hidden, 1)
        self.mask = nn.Conv2d(int(num_queries), hidden, 1)
        self.footprint = nn.Conv2d(1, hidden, 1)
        self.refine = ConvNormAct(hidden, hidden)
        self.output = nn.Conv2d(hidden, int(num_queries), 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        features: torch.Tensor,
        mask_logits: torch.Tensor,
        footprint: torch.Tensor,
    ) -> torch.Tensor:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        mask_logits = F.interpolate(
            mask_logits.float(),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = (
            self.feature(features.float())
            + self.mask(mask_logits)
            + self.footprint(footprint.float())
        )
        return mask_logits + self.output(self.refine(hidden))


class RasterInstanceControl(nn.Module):
    """Dense and instance-query room controls over one native raster encoder."""

    def __init__(
        self,
        taxonomy: FloorplanTaxonomy,
        *,
        decoder_channels: int = 128,
        hidden_dim: int = 128,
        num_queries: int = 24,
        layers: int = 3,
        heads: int = 4,
        full_resolution_masks: bool = False,
        full_resolution_channels: int = 32,
        query_existence_gate: bool = False,
        dual_control: bool = False,
        adaptive_query_refinement: bool = False,
        partition_consistent: bool = False,
        mask_pooled_class_fusion: bool = False,
        full_resolution_class_sampler: bool = False,
        owner_semantic_reassignment: bool = False,
        room_boundary_owner_refiner: bool = False,
        query_set_head: bool = False,
        query_set_hidden: int = 64,
        query_set_pool: int = 48,
        query_set_layers: int = 0,
        query_set_routing_attention: bool = False,
        query_set_route_router_count: int = 12,
        query_set_route_top_k: int = 4,
        joint_room_slot_state: bool = False,
        joint_slot_owner_fusion: bool = False,
        query_count_gate: bool = False,
        dense_identity_refiner: bool = False,
        room_program_head: bool = False,
        program_token_conditioning: bool = False,
        program_class_prior_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if query_set_head and joint_room_slot_state:
            raise ValueError(
                "query_set_head and joint_room_slot_state are mutually exclusive"
            )
        if joint_slot_owner_fusion and not joint_room_slot_state:
            raise ValueError(
                "joint_slot_owner_fusion requires joint_room_slot_state"
            )
        if program_token_conditioning and not room_program_head:
            raise ValueError(
                "program token conditioning requires room_program_head"
            )
        if program_class_prior_weight < 0.0 or (
            program_class_prior_weight > 0.0 and not room_program_head
        ):
            raise ValueError(
                "program class prior requires room_program_head"
            )
        self.room = ContourRoomSegmenter(
            len(taxonomy.room_ids),
            pretrained=False,
            decoder_channels=int(decoder_channels),
        )
        self.mask_decoder = RoomMaskDecoder(
            int(decoder_channels),
            len(taxonomy.room_ids),
            hidden_dim=int(hidden_dim),
            num_queries=int(num_queries),
            layers=int(layers),
            heads=int(heads),
            feature_scales=(1, 2, 4),
            program_token_dim=(
                int(hidden_dim) if program_token_conditioning else 0
            ),
            adaptive_query_refinement=bool(adaptive_query_refinement),
            partition_consistent=bool(partition_consistent),
        )
        self.full_resolution_masks = bool(full_resolution_masks)
        self.query_existence_gate = bool(query_existence_gate)
        self.dual_control = bool(dual_control)
        self.control_projection = None
        if self.dual_control:
            projection = nn.Conv2d(6, 3, 1)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.control_projection = projection
        self.mask_refiner = (
            QueryMaskRefiner(
                int(decoder_channels),
                int(num_queries),
                hidden_channels=int(full_resolution_channels),
            )
            if self.full_resolution_masks
            else None
        )
        self.mask_pooled_class_fusion = (
            MaskPooledClassFusion(
                int(hidden_dim),
                int(decoder_channels),
                len(taxonomy.room_ids),
            )
            if mask_pooled_class_fusion
            else None
        )
        self.full_resolution_class_sampler = (
            FullResolutionClassSampler(
                int(hidden_dim),
                int(decoder_channels),
                len(taxonomy.room_ids),
            )
            if full_resolution_class_sampler
            else None
        )
        self.owner_semantic_reassignment = (
            OwnerSemanticReassignment()
            if owner_semantic_reassignment
            else None
        )
        self.room_boundary_owner_refiner = (
            RoomBoundaryOwnerRefiner(
                int(decoder_channels),
                int(num_queries),
            )
            if room_boundary_owner_refiner
            else None
        )
        self.query_set_head = (
            RoomQuerySetHead(
                int(hidden_dim),
                int(decoder_channels),
                len(taxonomy.room_ids),
                int(num_queries),
                hidden_dim=int(query_set_hidden),
                pool_size=int(query_set_pool),
                attention_layers=int(query_set_layers),
                attention_heads=int(heads),
                routing_attention=bool(query_set_routing_attention),
                route_router_count=int(query_set_route_router_count),
                route_top_k=int(query_set_route_top_k),
            )
            if query_set_head
            else None
        )
        self.joint_room_slot_state = (
            RoomSlotStateHead(
                int(hidden_dim),
                int(decoder_channels),
                len(taxonomy.room_ids),
                int(num_queries),
            )
            if joint_room_slot_state
            else None
        )
        self.joint_slot_owner_fusion = bool(joint_slot_owner_fusion)
        self.query_count_gate = bool(query_count_gate)
        self.dense_identity_refiner = (
            PredictedRoomIdentityHead(
                int(decoder_channels),
                len(taxonomy.room_ids),
                use_image=True,
            )
            if dense_identity_refiner
            else None
        )
        self.room_program_head = (
            RoomProgramSetHead(
                int(decoder_channels),
                len(taxonomy.room_ids),
                hidden_dim=int(hidden_dim),
                tokens=6,
                layers=2,
                heads=int(heads),
                num_queries=int(num_queries),
            )
            if room_program_head
            else None
        )
        self.program_token_conditioning = bool(
            program_token_conditioning
        )
        self.program_class_prior_weight = float(program_class_prior_weight)

    def forward(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        control = control_image
        if self.dual_control:
            if control_image is None or control_image.shape[1] != 6:
                raise ValueError("dual control requires six image channels")
            native = control_image[:, :3]
            control = native + self.control_projection(control_image)
        hidden = self.room.encode_control(
            footprint,
            control_image=control,
        )
        dense_logits = self.room.final(hidden)
        if self.dense_identity_refiner is not None:
            dense_logits = self.dense_identity_refiner(
                hidden,
                dense_logits,
                image=control,
            )["room_logits"]
        program = (
            self.room_program_head(hidden)
            if self.room_program_head is not None
            else None
        )
        # Scale 1 of RoomMaskDecoder must remain the native control resolution;
        # the decoder already creates 2x and 4x pooled levels internally.
        instance_hidden = hidden
        instances = self.mask_decoder(
            instance_hidden,
            program_tokens=(
                program["class_tokens"]
                if program is not None
                and self.program_token_conditioning
                else None
            ),
            adaptive_features=hidden,
        )
        if self.mask_refiner is not None:
            instances["mask_logits"] = self.mask_refiner(
                hidden,
                instances["mask_logits"],
                footprint,
            )
        if self.room_boundary_owner_refiner is not None:
            (
                instances["mask_logits"],
                instances["room_boundary_logits"],
                owner_diagnostics,
            ) = self.room_boundary_owner_refiner(
                hidden,
                instances["mask_logits"],
            )
            instances["room_boundary_owner_diagnostics"] = owner_diagnostics
        if self.mask_pooled_class_fusion is not None:
            instances["class_logits"], pooled_diagnostics = (
                self.mask_pooled_class_fusion(
                    instances["query_features"],
                    instances["mask_logits"],
                    instances["class_logits"],
                    hidden,
                )
            )
            instances["mask_pooled_class_diagnostics"] = pooled_diagnostics
            instances["query_existence_logits"] = (
                query_existence_from_class_logits(
                    instances["class_logits"]
                )
            )
        if self.full_resolution_class_sampler is not None:
            instances["class_logits"], sampler_diagnostics = (
                self.full_resolution_class_sampler(
                    instances["query_features"],
                    instances["mask_logits"],
                    instances["class_logits"],
                    hidden,
                )
            )
            instances["full_resolution_class_sampler_diagnostics"] = (
                sampler_diagnostics
            )
            instances["query_existence_logits"] = (
                query_existence_from_class_logits(
                    instances["class_logits"]
                )
            )
        if self.joint_room_slot_state is not None:
            instances.update(
                self.joint_room_slot_state(
                    hidden,
                    instances["query_features"],
                    instances["mask_logits"],
                    instances["class_logits"],
                )
            )
        if program is not None and self.program_class_prior_weight > 0.0:
            instances["class_logits"] = apply_program_class_prior(
                instances["class_logits"],
                program["presence_logits"],
                scale=self.program_class_prior_weight,
            )
            instances["query_existence_logits"] = (
                query_existence_from_class_logits(
                    instances["class_logits"]
                )
            )
            instances["program_class_prior_weight"] = torch.tensor(
                self.program_class_prior_weight,
                device=instances["class_logits"].device,
            )
        if self.query_set_head is not None:
            instances.update(
                self.query_set_head(
                    hidden,
                    instances["query_features"],
                    instances["mask_logits"],
                    instances["query_existence_logits"],
                )
            )
            if (
                self.query_count_gate
                and "query_count_candidate_mask" in instances
            ):
                instances["query_count_selection_mask"] = (
                    instances["query_count_soft_selection"]
                    if self.training
                    and "query_count_soft_selection" in instances
                    else instances["query_count_candidate_mask"]
                )
        if program is not None:
            instances.update(
                {
                    "room_program_class_logits": program["class_logits"],
                    "room_program_presence_logits": program[
                        "presence_logits"
                    ],
                    "room_program_cardinality_logits": program[
                        "cardinality_logits"
                    ],
                }
            )
        if self.owner_semantic_reassignment is not None:
            fused, reassignment_diagnostics = (
                self.owner_semantic_reassignment(
                    dense_logits,
                    instances["mask_logits"],
                    footprint,
                    instances.get("query_existence_logits"),
                )
            )
            instances["owner_semantic_reassignment_diagnostics"] = (
                reassignment_diagnostics
            )
        else:
            fused = fused_room_mask_logits(
                instances,
                dense_logits.shape[1],
                fusion_mode=(
                    "slot_owner"
                    if self.joint_slot_owner_fusion
                    else "sigmoid_existence"
                    if self.query_existence_gate
                    else "sigmoid"
                ),
            )
            fused = F.interpolate(
                fused,
                size=dense_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        instances.update(
            room_owner_readout(
                instances,
                dense_logits.shape[1],
                footprint=footprint,
                fallback_logits=fused,
            )
        )
        return {
            "room_logits": fused,
            "dense_logits": dense_logits,
            "fused_logits": fused,
            **instances,
        }

    def module_contract(self) -> dict[str, object]:
        return raster_instance_module_contract(self)


def blend_room_instance_probabilities(
    outputs: dict[str, torch.Tensor],
    weight: float,
) -> torch.Tensor:
    """Blend dense and query-room probabilities with the training contract."""

    dense = outputs["dense_logits"].softmax(dim=1)
    fused = outputs["fused_logits"].softmax(dim=1)
    return (1.0 - float(weight)) * dense + float(weight) * fused


__all__ = [
    "DeepLabRasterRoomControl", "QueryMaskRefiner", "RasterControlBank",
    "RasterDenseQueryRoomBlend", "RasterInstanceClassRoutedAdapter",
    "RasterInstanceControl", "RasterInstanceRoomAdapter",
    "RasterInstanceSoftRoutedAdapter", "blend_room_instance_probabilities",
    "build_raster_room_model", "load_raster_control_bank",
    "load_raster_instance_control_model", "load_raster_room_model",
]

from floorflow.heads.raster_control_bank_loading import (
    load_raster_control_bank,
    load_raster_instance_control_model,
    load_raster_room_model,
)
from floorflow.heads.raster_room_model_build import build_raster_room_model

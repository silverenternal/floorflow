"""Detection-first room-aware two-level polygon decoder."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.polygon_loop import PolygonLoopHead
from floorflow.heads.raster_control_bank import ContourRoomSegmenter


class RoomAwarePolygonHead(PolygonLoopHead):
    """PolyRoom/CAGE-inspired polygon and corner queries over dense room evidence."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_dim: int = 128,
        num_queries: int = 32,
        points: int = 24,
        layers: int = 2,
        heads: int = 4,
        class_token_dim: int = 96,
    ) -> None:
        super().__init__(
            in_channels,
            num_room_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            points=points,
            layers=layers,
            heads=heads,
            class_token_dim=class_token_dim,
        )
        self.query_feature = nn.Linear(in_channels, hidden_dim)
        self.support_memory = nn.Conv2d(num_room_classes, hidden_dim, 1)
        self.count_head = nn.Linear(hidden_dim, max(1, points - 2))

    def _dense_query_prior(
        self,
        features: torch.Tensor,
        room_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_grid = F.adaptive_avg_pool2d(features, (16, 16))
        feature_tokens = feature_grid.flatten(2).transpose(1, 2)
        probabilities = F.interpolate(
            room_logits.softmax(dim=1),
            size=(16, 16),
            mode="bilinear",
            align_corners=False,
        )
        scores = probabilities.max(dim=1).values.flatten(1)
        indices = torch.topk(
            scores,
            k=self.queries.shape[1],
            dim=1,
        ).indices
        selected = torch.gather(
            feature_tokens,
            1,
            indices[..., None].expand(-1, -1, feature_tokens.shape[-1]),
        )
        memory = self.support_memory(probabilities).flatten(2).transpose(1, 2)
        return self.query_feature(selected), memory

    def forward(
        self,
        features: torch.Tensor,
        room_logits: torch.Tensor,
        class_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query_prior, support_memory = self._dense_query_prior(
            features,
            room_logits,
        )
        prediction = super().forward(
            features,
            class_tokens=class_tokens,
            query_prior=query_prior,
            extra_memory=support_memory,
        )
        prediction["corner_count_logits"] = self.count_head(
            prediction["query_hidden"]
        )
        return prediction


class RasterPolygonControl(nn.Module):
    """Native raster encoder with a detection-first polygon decoder."""

    def __init__(
        self,
        taxonomy: FloorplanTaxonomy,
        *,
        decoder_channels: int = 128,
        hidden_dim: int = 128,
        num_queries: int = 32,
        points: int = 24,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.room = ContourRoomSegmenter(
            len(taxonomy.room_ids),
            pretrained=False,
            decoder_channels=int(decoder_channels),
        )
        self.polygon_head = RoomAwarePolygonHead(
            int(decoder_channels),
            len(taxonomy.room_ids),
            hidden_dim=int(hidden_dim),
            num_queries=int(num_queries),
            points=int(points),
            layers=int(layers),
            heads=int(heads),
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
        dense_logits = self.room.final(hidden)
        polygon = self.polygon_head(hidden, dense_logits)
        return {
            "room_logits": dense_logits,
            "dense_logits": dense_logits,
            **polygon,
        }


@torch.inference_mode()
def decode_polygons_with_dense_classes(
    predictions: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    height: int = 256,
    width: int = 256,
    validity_threshold: float = 0.0,
) -> torch.Tensor:
    """Decode polygon geometry with class evidence from dense room logits."""

    polygons = predictions["polygons"].detach().cpu().numpy()
    valid = predictions["valid_logits"].detach().cpu().numpy()
    dense = predictions["dense_logits"].detach().float().cpu().numpy()
    output = np.zeros((polygons.shape[0], height, width), dtype=np.uint16)
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    for batch_index in range(polygons.shape[0]):
        candidates = []
        for query in range(polygons.shape[1]):
            if valid[batch_index, query] <= float(validity_threshold):
                continue
            points = np.rint(
                polygons[batch_index, query]
                * np.asarray([width - 1, height - 1], dtype=np.float32)
            ).astype(np.int32)
            area = abs(float(cv2.contourArea(points)))
            if area <= 0.0:
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            selected = mask.astype(bool)
            if not selected.any():
                continue
            class_index = int(
                dense[batch_index]
                .reshape(dense.shape[1], -1)[:, selected.reshape(-1)]
                .mean(axis=1)
                .argmax()
            )
            candidates.append(
                (
                    area,
                    query,
                    points,
                    int(room_ids[class_index]),
                )
            )
        for _area, _query, points, class_id in sorted(
            candidates,
            reverse=True,
        ):
            cv2.fillPoly(
                output[batch_index],
                [points],
                color=class_id,
            )
    return torch.from_numpy(output.astype(np.int64)).to(
        predictions["polygons"].device
    )


__all__ = [
    "RasterPolygonControl",
    "RoomAwarePolygonHead",
    "decode_polygons_with_dense_classes",
]

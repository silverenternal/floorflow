"""Adapters for evaluating Raster2Seq on FloorFlow edit pairs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy

R2S_TO_CLASS_NAME = {
    0: None,
    1: "room:Kitchen",
    2: "room:LivingRoom",
    3: "room:Bedroom",
    4: "room:Bath",
    5: "room:Entry",
    6: "room:Storage",
    7: "room:Garage",
    8: "room:Undefined",
    9: "window:Regular",
    10: "door:Swing Beside",
    11: None,
}

R2S_SET3_COLORS = {
    0: (141, 211, 199),
    1: (255, 255, 179),
    2: (190, 186, 218),
    3: (251, 128, 114),
    4: (128, 177, 211),
    5: (253, 180, 98),
    6: (179, 222, 105),
    7: (252, 205, 229),
    8: (217, 217, 217),
}


@dataclass(frozen=True)
class Raster2SeqProposal:
    polygon: np.ndarray
    class_id: int
    class_probability: float
    area: float


def _class_id(taxonomy: FloorplanTaxonomy, name: str | None) -> int | None:
    if name is None:
        return None
    try:
        return taxonomy.class_id(name)
    except KeyError:
        return None


def render_raster2seq_input(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    """Render a neutral architectural line drawing from exclusive control labels."""
    if labels.ndim != 2:
        raise ValueError("labels must be a two-dimensional label map")
    labels = labels.astype(np.int64, copy=False)
    palette = np.asarray(
        [
            np.rint(
                0.3 * np.asarray(taxonomy.color(index), dtype=np.float32)
                + 0.7 * 255.0
            ).astype(np.uint8)
            for index in range(taxonomy.num_classes)
        ],
        dtype=np.uint8,
    )
    image = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    room_mask = np.isin(labels, taxonomy.room_ids)
    image[room_mask] = palette[labels[room_mask]]

    furniture_names = ("furniture",)
    furniture_ids = [
        index
        for index, name in enumerate(taxonomy.class_names)
        if name.startswith(furniture_names)
    ]
    if furniture_ids:
        furniture = np.isin(labels, furniture_ids)
        image[furniture] = (215, 215, 215)

    door_ids = [
        index
        for index, name in enumerate(taxonomy.class_names)
        if name.startswith("door:")
    ]
    window_ids = [
        index
        for index, name in enumerate(taxonomy.class_names)
        if name.startswith("window:")
    ]
    if door_ids:
        image[np.isin(labels, door_ids)] = (60, 60, 180)
    if window_ids:
        image[np.isin(labels, window_ids)] = (190, 110, 40)

    wall_id = taxonomy.class_id("wall")
    image[labels == wall_id] = (0, 0, 0)

    boundary = np.zeros_like(room_mask, dtype=bool)
    boundary[:, 1:] |= room_mask[:, 1:] & (labels[:, 1:] != labels[:, :-1])
    boundary[:, :-1] |= room_mask[:, :-1] & (labels[:, :-1] != labels[:, 1:])
    boundary[1:, :] |= room_mask[1:, :] & (labels[1:, :] != labels[:-1, :])
    boundary[:-1, :] |= room_mask[:-1, :] & (labels[:-1, :] != labels[1:, :])
    image[boundary] = (0, 0, 0)

    inside = labels > 0
    footprint_edge = np.zeros_like(inside, dtype=bool)
    footprint_edge[:, 1:] |= inside[:, 1:] != inside[:, :-1]
    footprint_edge[:, :-1] |= inside[:, :-1] != inside[:, 1:]
    footprint_edge[1:, :] |= inside[1:, :] != inside[:-1, :]
    footprint_edge[:-1, :] |= inside[:-1, :] != inside[1:, :]
    image[footprint_edge] = (0, 0, 0)
    return image


def render_raster2seq_official_style(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    """Render CubiCasa labels with Raster2Seq's paper-style color/outline scheme."""

    if labels.ndim != 2:
        raise ValueError("labels must be a two-dimensional label map")
    labels = labels.astype(np.int64, copy=False)
    image = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    for taxonomy_id, class_name in enumerate(taxonomy.class_names):
        raster2seq_id = next(
            (
                key
                for key, value in R2S_TO_CLASS_NAME.items()
                if value == class_name
            ),
            None,
        )
        if raster2seq_id is None:
            continue
        mask = (labels == taxonomy_id).astype(np.uint8)
        if not mask.any():
            continue
        is_opening = class_name.startswith(("door:", "window:"))
        if is_opening:
            color = (255, 0, 0) if class_name.startswith("door:") else (0, 0, 255)
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
            continue
        color = R2S_SET3_COLORS.get(raster2seq_id)
        if color is None:
            continue
        alpha_image = np.full_like(image, color)
        mask_3d = mask.astype(bool)
        image[mask_3d] = (
            0.6 * alpha_image[mask_3d] + 0.4 * image[mask_3d]
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(image, contours, -1, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def _sequence_segments(
    sequence: Sequence[object],
) -> list[tuple[int, list[np.ndarray]]]:
    segments: list[tuple[int, list[np.ndarray]]] = []
    current: list[np.ndarray] = []
    start = 0
    separators = {1, 2, 3, -1}
    for index, token in enumerate(sequence):
        if isinstance(token, (int, np.integer)):
            if int(token) in separators and current:
                segments.append((start, current))
                current = []
            start = index + 1
            if int(token) in {2, -1}:
                break
            continue
        if not current:
            start = index
        point = np.asarray(token, dtype=np.float32)
        if point.shape != (2,):
            raise ValueError(f"unexpected corner token shape: {point.shape}")
        current.append(point)
    if current:
        segments.append((start, current))
    return segments


def decode_raster2seq_proposals(
    sequence: Sequence[object],
    room_logits: torch.Tensor | np.ndarray,
    taxonomy: FloorplanTaxonomy,
    image_size: int = 256,
) -> list[Raster2SeqProposal]:
    """Decode one token sequence into typed proposals with class confidence."""

    logits = (
        room_logits.detach().cpu().numpy()
        if torch.is_tensor(room_logits)
        else np.asarray(room_logits)
    )
    if len(sequence) != len(logits):
        raise ValueError("sequence and class logits must have equal length")

    proposals: list[Raster2SeqProposal] = []
    for start, segment in _sequence_segments(sequence):
        count = len(segment)
        segment_logits = logits[start : start + count]
        if not len(segment_logits):
            continue
        final_logits = np.asarray(segment_logits[-1], dtype=np.float64)
        shifted = final_logits - final_logits.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        r2s_class = int(np.argmax(final_logits))
        class_id = _class_id(taxonomy, R2S_TO_CLASS_NAME.get(r2s_class))
        if class_id is None:
            continue
        corners = np.rint(np.asarray(segment) * (image_size - 1)).astype(np.int32)
        corners = np.clip(corners, 0, image_size - 1)
        if len(corners) < 2:
            continue
        area = abs(float(cv2.contourArea(corners)))
        is_opening = taxonomy.class_names[class_id].startswith(("door:", "window:"))
        keep = (
            (is_opening and len(corners) == 2)
            or (not is_opening and len(corners) >= 3 and area >= 100.0)
        )
        if keep:
            proposals.append(
                Raster2SeqProposal(
                    polygon=corners,
                    class_id=int(class_id),
                    class_probability=float(probabilities[r2s_class]),
                    area=float(area),
                )
            )
    return proposals


def decode_raster2seq_sequence(
    sequence: Sequence[object],
    room_logits: torch.Tensor | np.ndarray,
    taxonomy: FloorplanTaxonomy,
    image_size: int = 256,
) -> tuple[list[np.ndarray], list[int]]:
    """Decode one Raster2Seq token sequence into polygons and CubiCasa labels."""

    proposals = decode_raster2seq_proposals(
        sequence,
        room_logits,
        taxonomy,
        image_size=image_size,
    )
    return (
        [proposal.polygon for proposal in proposals],
        [proposal.class_id for proposal in proposals],
    )


def render_raster2seq_predictions(
    polygons: Iterable[np.ndarray],
    labels: Iterable[int],
    shape: tuple[int, int],
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    """Render predicted room polygons and openings into an exclusive label raster."""
    output = np.zeros(shape, dtype=np.uint8)
    rooms: list[tuple[np.ndarray, int]] = []
    openings: list[tuple[np.ndarray, int]] = []
    for polygon, class_id in zip(polygons, labels, strict=True):
        target = (
            openings
            if taxonomy.class_names[int(class_id)].startswith(("door:", "window:"))
            else rooms
        )
        target.append((polygon.astype(np.int32), int(class_id)))

    for polygon, class_id in sorted(rooms, key=lambda item: -cv2.contourArea(item[0])):
        cv2.fillPoly(output, [polygon], color=int(class_id))
    for polygon, class_id in openings:
        if len(polygon) == 2:
            cv2.line(
                output,
                tuple(int(value) for value in polygon[0]),
                tuple(int(value) for value in polygon[1]),
                color=int(class_id),
                thickness=2,
            )
        else:
            cv2.polylines(output, [polygon], True, color=int(class_id), thickness=2)
    return output


__all__ = [
    "R2S_SET3_COLORS",
    "R2S_TO_CLASS_NAME",
    "Raster2SeqProposal",
    "decode_raster2seq_proposals",
    "decode_raster2seq_sequence",
    "render_raster2seq_input",
    "render_raster2seq_official_style",
    "render_raster2seq_predictions",
]

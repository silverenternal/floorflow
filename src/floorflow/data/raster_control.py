"""Render non-room reconstruction inputs without exposing room identities."""

from __future__ import annotations

import warnings
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from floorflow.data.taxonomy import FloorplanTaxonomy


def _open_image(path, mode: str) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        return Image.open(path).convert(mode)


def raster_control_image(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    """Return a three-channel architectural drawing from non-room evidence.

    Channel 0 contains walls, doors, windows, and structure. Channel 1 contains
    fixed furniture. Channel 2 contains the footprint and a dilated structural
    band. Room-class labels are never read.
    """

    labels = np.asarray(labels)
    footprint = np.asarray(footprint, dtype=bool)
    if labels.ndim != 2 or footprint.shape != labels.shape:
        raise ValueError("labels and footprint must share one two-dimensional shape")

    structure = np.zeros_like(labels, dtype=bool)
    structure |= labels == taxonomy.class_id("wall")
    structure |= np.isin(labels, np.asarray(taxonomy.opening_ids))
    structure |= labels == taxonomy.class_id("structure")
    furniture = np.isin(labels, np.asarray(taxonomy.furniture_ids))

    structural_mask = structure.astype(np.uint8)
    structural_band = cv2.dilate(
        structural_mask,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    footprint_uint8 = footprint.astype(np.uint8)
    contours, _ = cv2.findContours(
        footprint_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boundary = np.zeros_like(footprint_uint8)
    if contours:
        cv2.drawContours(boundary, contours, -1, 1, thickness=2)

    image = np.ones((*labels.shape, 3), dtype=np.float32)
    image[:, :, 0] = 1.0 - (structure | structural_band).astype(np.float32)
    image[:, :, 1] = np.where(furniture, 0.55, 1.0).astype(np.float32)
    image[:, :, 2] = 1.0 - (
        boundary.astype(bool) | ~footprint
    ).astype(np.float32)
    return np.moveaxis(image, -1, 0)


def non_room_control_labels(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    """Remove every room-class label while preserving non-room structure."""

    values = np.asarray(labels).copy()
    footprint_mask = np.asarray(footprint, dtype=bool)
    if values.shape != footprint_mask.shape:
        raise ValueError("labels and footprint must share one shape")
    values[np.isin(values, np.asarray(taxonomy.room_ids, dtype=np.int64))] = 0
    values[~footprint_mask] = 0
    return values


def image_normalization(
    image: np.ndarray,
    *,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    mean_array = np.asarray(mean, dtype=np.float32)[:, None, None]
    std_array = np.asarray(std, dtype=np.float32)[:, None, None]
    return (array - mean_array) / std_array


def load_cubicasa_raster(path, image_size: int = 256) -> np.ndarray:
    """Load the native CubiCasa floorplan raster as normalized RGB channels."""

    image = _open_image(path, "RGB")
    if image.size != (int(image_size), int(image_size)):
        image = image.resize(
            (int(image_size), int(image_size)),
            Image.Resampling.BILINEAR,
        )
    array = np.moveaxis(
        np.asarray(image, dtype=np.float32) / 255.0,
        -1,
        0,
    )
    return image_normalization(array)


def load_cubicasa_rgb_raster(path, image_size: int = 256) -> np.ndarray:
    """Load the native CubiCasa raster as unnormalized RGB [0,1]."""

    image = _open_image(path, "RGB")
    if image.size != (int(image_size), int(image_size)):
        image = image.resize(
            (int(image_size), int(image_size)),
            Image.Resampling.BILINEAR,
        )
    return np.asarray(image, dtype=np.float32) / 255.0


@lru_cache(maxsize=256)
def cubicasa_floor_crops(path, floor_count: int) -> list[tuple[int, int, int, int]]:
    """Detect page-level floor drawing bounds without room-label supervision."""

    count = int(floor_count)
    if count <= 0:
        raise ValueError("floor_count must be positive")
    gray = np.asarray(_open_image(path, "L"), dtype=np.uint8)
    height, width = gray.shape
    ink = (gray < 248).astype(np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    best_boxes: list[tuple[int, int, int, int]] = []
    best_error = 1_000_000
    for kernel_size in (3, 5, 7, 9, 11, 15):
        joined = cv2.dilate(
            ink,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        )
        component_count, _labels, stats, _centroids = (
            cv2.connectedComponentsWithStats(joined, connectivity=8)
        )
        boxes = []
        for component in range(1, int(component_count)):
            left, top, box_width, box_height, area = (
                int(value) for value in stats[component]
            )
            fraction = area / max(1, box_width * box_height)
            if (
                box_width >= 0.15 * width
                and box_height >= 0.15 * height
                and area >= 0.015 * width * height
                and fraction >= 0.03
            ):
                margin = max(8, int(0.005 * max(width, height)))
                boxes.append(
                    (
                        max(0, left - margin),
                        max(0, top - margin),
                        min(width, left + box_width + margin),
                        min(height, top + box_height + margin),
                    )
                )
        boxes.sort(key=lambda box: box[0])
        error = abs(len(boxes) - count)
        if error < best_error:
            best_boxes, best_error = boxes, error
        if len(boxes) == count:
            return boxes
        if count == 1 and boxes:
            return [
                (
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                )
            ]
    if best_boxes:
        return best_boxes
    raise ValueError(f"detected no floor drawing regions, expected {count}")


def load_cubicasa_floor_rgb(
    path,
    *,
    floor_index: int,
    floor_count: int,
    target_center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Load one page-level CubiCasa floor drawing as target-free RGB."""

    boxes = cubicasa_floor_crops(path, floor_count)
    index = int(floor_index)
    if not 0 <= index < int(floor_count):
        raise IndexError("floor index is outside the detected page plan")
    image = _open_image(path, "RGB")
    if len(boxes) == int(floor_count):
        box = boxes[index]
    elif target_center is not None:
        box = min(
            boxes,
            key=lambda item: (
                (item[0] + item[2]) / 2.0 / image.width - target_center[0]
            )
            ** 2
            + (
                (item[1] + item[3]) / 2.0 / image.height - target_center[1]
            )
            ** 2,
        )
    elif len(boxes) == 1 and int(floor_count) > 1:
        left, top, right, bottom = boxes[0]
        if right - left >= bottom - top:
            width = (right - left) / float(floor_count)
            box = (
                round(left + index * width),
                top,
                round(left + (index + 1) * width),
                bottom,
            )
        else:
            height = (bottom - top) / float(floor_count)
            box = (
                left,
                round(top + index * height),
                right,
                round(top + (index + 1) * height),
            )
    else:
        box = boxes[index]
    return np.asarray(image.crop(box), dtype=np.uint8)


def footprint_crop_box(
    footprint: np.ndarray,
    *,
    margin_fraction: float = 0.01,
) -> tuple[int, int, int, int]:
    """Return one content box for a non-empty floorplan mask."""

    ys, xs = np.nonzero(np.asarray(footprint, dtype=bool))
    if not len(xs):
        raise ValueError("footprint is empty")
    height, width = footprint.shape
    margin = max(2, round(float(margin_fraction) * max(height, width)))
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + margin + 1),
        min(height, int(ys.max()) + margin + 1),
    )


def footprint_center(footprint: np.ndarray) -> tuple[float, float]:
    """Return the normalized center of a non-empty floorplan mask."""

    left, top, right, bottom = footprint_crop_box(
        footprint,
        margin_fraction=0.0,
    )
    height, width = np.asarray(footprint).shape
    return (
        0.5 * (left + right - 1) / max(1, width - 1),
        0.5 * (top + bottom - 1) / max(1, height - 1),
    )


def _fit_square(
    array: np.ndarray,
    size: int,
    *,
    resample: Image.Resampling,
    fill: int | tuple[int, int, int],
) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    scale = min(float(size) / image.width, float(size) / image.height)
    resized = image.resize(
        (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        ),
        resample,
    )
    if resized.mode == "RGB":
        canvas = Image.new("RGB", (size, size), fill)  # type: ignore[arg-type]
    else:
        canvas = Image.new("L", (size, size), int(fill))
    canvas.paste(
        resized,
        ((size - resized.width) // 2, (size - resized.height) // 2),
    )
    return np.asarray(canvas, dtype=np.uint8)


def crop_floor_masks(
    target: np.ndarray,
    footprint: np.ndarray,
    *,
    image_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop full-page room targets to one aligned square floor frame."""

    left, top, right, bottom = footprint_crop_box(footprint)
    target_crop = np.asarray(target)[top:bottom, left:right]
    footprint_crop = np.asarray(footprint)[top:bottom, left:right]
    return (
        _fit_square(
            target_crop,
            image_size,
            resample=Image.Resampling.NEAREST,
            fill=0,
        ).astype(np.int64),
        _fit_square(
            footprint_crop,
            image_size,
            resample=Image.Resampling.NEAREST,
            fill=0,
        ).astype(np.float32),
    )


def fit_channel_raster(
    array: np.ndarray,
    *,
    image_size: int = 256,
    fill: float = 0.0,
) -> np.ndarray:
    """Resize a channel-first raster with aspect-preserving padding."""

    values = np.asarray(array, dtype=np.float32)
    channels, height, width = values.shape
    scale = min(float(image_size) / height, float(image_size) / width)
    resized_h = max(1, round(height * scale))
    resized_w = max(1, round(width * scale))
    resized = np.stack(
        [
            cv2.resize(
                values[index],
                (resized_w, resized_h),
                interpolation=cv2.INTER_LINEAR,
            )
            for index in range(channels)
        ]
    )
    output = np.full(
        (channels, image_size, image_size),
        float(fill),
        dtype=np.float32,
    )
    top = (image_size - resized_h) // 2
    left = (image_size - resized_w) // 2
    output[:, top : top + resized_h, left : left + resized_w] = resized
    return output


def load_cubicasa_floor_raster(
    path,
    *,
    floor_index: int,
    floor_count: int,
    target_center: tuple[float, float] | None = None,
    image_size: int = 256,
) -> np.ndarray:
    """Load one page-level floor as normalized square RGB channels."""

    image = load_cubicasa_floor_rgb(
        path,
        floor_index=floor_index,
        floor_count=floor_count,
        target_center=target_center,
    )
    array = _fit_square(
        image,
        image_size,
        resample=Image.Resampling.BILINEAR,
        fill=(255, 255, 255),
    )
    return image_normalization(
        np.moveaxis(array.astype(np.float32) / 255.0, -1, 0)
    )


def _cubicasa_line_components(
    path,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = _open_image(path, "RGB")
    if image.size != (int(image_size), int(image_size)):
        image = image.resize(
            (int(image_size), int(image_size)),
            Image.Resampling.BILINEAR,
        )
    rgb = np.asarray(image, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(smooth, 40, 120)
    adaptive = cv2.adaptiveThreshold(
        smooth,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        7,
    )
    ink = ((edges > 0) | (adaptive > 0)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        ink,
        connectivity=8,
    )
    cleaned = np.zeros_like(ink)
    for component in range(1, int(count)):
        area = int(stats[component, cv2.CC_STAT_AREA])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        aspect = max(width, height) / max(1, min(width, height))
        if (
            area >= 24
            and max(width, height) >= 12
            and (aspect >= 2.5 or area >= 120)
        ):
            cleaned[components == component] = 1
    saturation = rgb.max(axis=-1).astype(np.int16) - rgb.min(axis=-1)
    color_edges = cv2.Canny(saturation.astype(np.uint8), 12, 35)
    footprint = (gray < 250).astype(np.uint8)
    return cleaned, edges, color_edges, footprint


def load_cubicasa_line_raster(path, image_size: int = 256) -> np.ndarray:
    """Extract a structural line drawing from a native CubiCasa image."""

    cleaned, edges, color_edges, footprint = _cubicasa_line_components(
        path,
        image_size,
    )
    channels = np.stack(
        [
            1.0 - cleaned.astype(np.float32),
            1.0 - np.maximum(edges, color_edges).astype(np.float32) / 255.0,
            1.0 - footprint.astype(np.float32),
        ]
    )
    return image_normalization(channels)


def load_cubicasa_structure_mask(path, image_size: int = 256) -> np.ndarray:
    """Return native drawing ink used to separate room components."""

    cleaned, _edges, _color_edges, _footprint = _cubicasa_line_components(
        path,
        image_size,
    )
    return cleaned > 0


__all__ = [
    "crop_floor_masks",
    "cubicasa_floor_crops",
    "fit_channel_raster",
    "footprint_center",
    "footprint_crop_box",
    "image_normalization",
    "load_cubicasa_floor_raster",
    "load_cubicasa_floor_rgb",
    "load_cubicasa_line_raster",
    "load_cubicasa_raster",
    "load_cubicasa_rgb_raster",
    "load_cubicasa_structure_mask",
    "non_room_control_labels",
    "raster_control_image",
]

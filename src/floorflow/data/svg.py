"""CubiCasa SVG to semantic-floorplan raster conversion.

The parser intentionally uses only the standard XML parser, NumPy, and Pillow.
It does not import the original CubiCasa training code or its LSTM dependency.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from floorflow.data.classes import (
    furniture_class,
    room_class,
)
from floorflow.data.taxonomy import STRUCTURE_IDS, FloorplanTaxonomy

_TRANSFORM_RE = re.compile(r"([A-Za-z]+)\(([^)]*)\)")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _numbers(value: str) -> list[float]:
    return [float(part) for part in re.split(r"[\s,]+", value.strip()) if part]


def parse_transform(value: str | None) -> np.ndarray:
    """Parse the SVG transform subset used by CubiCasa."""

    matrix = np.eye(3, dtype=np.float64)
    if not value:
        return matrix
    for name, raw_args in _TRANSFORM_RE.findall(value):
        args = _numbers(raw_args)
        local = np.eye(3, dtype=np.float64)
        if name == "matrix" and len(args) == 6:
            a, b, c, d, e, f = args
            local = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
        elif name == "translate" and len(args) >= 1:
            local[0, 2] = args[0]
            local[1, 2] = args[1] if len(args) > 1 else 0.0
        elif name == "scale" and len(args) >= 1:
            local[0, 0] = args[0]
            local[1, 1] = args[1] if len(args) > 1 else args[0]
        elif name == "rotate" and len(args) >= 1:
            angle = math.radians(args[0])
            rotation = np.array(
                [
                    [math.cos(angle), -math.sin(angle), 0.0],
                    [math.sin(angle), math.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            if len(args) == 3:
                cx, cy = args[1], args[2]
                to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
                back = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
                rotation = back @ rotation @ to_origin
            local = rotation
        matrix = matrix @ local
    return matrix


def _parse_points(value: str | None) -> np.ndarray:
    values = _numbers(value or "")
    if len(values) < 6 or len(values) % 2:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(-1, 2)


def _parse_path_points(value: str | None) -> np.ndarray:
    """Approximate SVG paths by the endpoints of each line/curve segment."""

    if not value:
        return np.empty((0, 2), dtype=np.float64)
    value = value.replace("NaN", "").replace("nan", "")
    tokens = re.findall(
        r"[MmLlHhVvQqTtCcSsAaZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?",
        value,
    )
    parameter_counts = {
        "M": 2,
        "L": 2,
        "H": 1,
        "V": 1,
        "Q": 4,
        "T": 2,
        "C": 6,
        "S": 4,
        "A": 7,
        "Z": 0,
    }
    points: list[tuple[float, float]] = []
    index = 0
    raw_command = "M"
    current = np.zeros(2, dtype=np.float64)
    start = current.copy()
    while index < len(tokens):
        if tokens[index].isalpha():
            raw_command = tokens[index]
            index += 1
            if raw_command.upper() == "Z":
                current = start.copy()
                continue
        command = raw_command.upper()
        count = parameter_counts.get(command)
        if count is None or index + count > len(tokens):
            break
        values = [float(token) for token in tokens[index : index + count]]
        index += count
        if count == 0:
            continue
        if command == "H":
            target = np.asarray([values[0], current[1]], dtype=np.float64)
        elif command == "V":
            target = np.asarray([current[0], values[0]], dtype=np.float64)
        elif count >= 2:
            target = np.asarray([values[-2], values[-1]], dtype=np.float64)
        else:
            continue
        if raw_command.islower():
            target = current + target
        current = target
        if command == "M":
            start = current.copy()
            raw_command = "l" if raw_command.islower() else "L"
        points.append((float(current[0]), float(current[1])))
    return np.asarray(points, dtype=np.float64) if len(points) >= 2 else np.empty((0, 2))


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homogeneous @ matrix.T)[:, :2]


def _direct_polygon_points(element: ET.Element, parent_matrix: np.ndarray) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for child in element:
        tag = _local_name(child.tag)
        if tag not in {"polygon", "path"}:
            continue
        points = _parse_points(child.get("points")) if tag == "polygon" else _parse_path_points(child.get("d"))
        matrix = parent_matrix @ parse_transform(child.get("transform"))
        if len(points):
            polygons.append(_transform_points(points, matrix))
    return polygons


def _furniture_boundary_points(element: ET.Element, parent_matrix: np.ndarray) -> list[np.ndarray]:
    """Read only CubiCasa's furniture BoundaryPolygon, not its internal details."""

    polygons: list[np.ndarray] = []
    for child in element:
        if _local_name(child.tag) != "g" or "BoundaryPolygon" not in (child.get("class") or "").split():
            continue
        boundary_matrix = parent_matrix @ parse_transform(child.get("transform"))
        for shape in child:
            tag = _local_name(shape.tag)
            if tag not in {"polygon", "path"}:
                continue
            points = _parse_points(shape.get("points")) if tag == "polygon" else _parse_path_points(shape.get("d"))
            matrix = boundary_matrix @ parse_transform(shape.get("transform"))
            if len(points):
                polygons.append(_transform_points(points, matrix))
    return polygons


def _draw_polygons(
    draw: ImageDraw.ImageDraw,
    polygons: list[np.ndarray],
    value: int,
    width: int,
    height: int,
) -> None:
    for points in polygons:
        coords = [(float(x), float(y)) for x, y in points]
        if len(coords) >= 3:
            draw.polygon(coords, fill=int(value))


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes enclosed by an arbitrary painted footprint."""

    if mask.size == 0 or not mask.any():
        return mask
    height, width = mask.shape
    padded = np.zeros((height + 4, width + 4), dtype=np.uint8)
    padded[2 : 2 + height, 2 : 2 + width] = mask.astype(np.uint8)
    # Pillow's fromarray can retain a read-only view and silently make floodfill
    # a no-op on some versions; materialize a writable image first.
    image = Image.fromarray(np.ascontiguousarray(padded * 255), mode="L").copy()
    ImageDraw.floodfill(image, (0, 0), 128)
    flooded = np.asarray(image, dtype=np.uint8)
    outside = flooded == 128
    filled = (~outside).astype(np.uint8)
    return filled[2 : 2 + height, 2 : 2 + width]


def _floor_nodes(root: ET.Element, parent_matrix: np.ndarray) -> list[tuple[str, np.ndarray]]:
    floors: list[tuple[str, np.ndarray]] = []
    for child in root:
        matrix = parent_matrix @ parse_transform(child.get("transform"))
        element_id = child.get("id") or ""
        if element_id.startswith("Floor-"):
            floors.append((element_id, parent_matrix))
            continue
        floors.extend(_floor_nodes(child, matrix))
    return floors


def list_cubicasa_floors(svg_path: str | Path) -> list[str]:
    root = ET.parse(svg_path).getroot()
    view_box = _numbers(root.get("viewBox") or "")
    if len(view_box) != 4:
        identity = np.eye(3, dtype=np.float64)
    else:
        min_x, min_y, width, height = view_box
        image_size = 256
        margin = 4
        scale = max(1.0, image_size - 2 * margin) / max(width, height)
        identity = np.array(
            [[scale, 0.0, margin - min_x * scale], [0.0, scale, margin - min_y * scale], [0.0, 0.0, 1.0]]
        )
    floors = _floor_nodes(root, identity)
    if floors:
        return [floor_id for floor_id, _matrix in floors]
    return ["Floor-1"]


def rasterize_cubicasa_svg(
    svg_path: str | Path,
    *,
    image_size: int = 256,
    output_shape: tuple[int, int] | None = None,
    floor_id: str | int | None = None,
    taxonomy: FloorplanTaxonomy | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one model.svg into target labels and an exterior footprint.

    Returns:
        target: ``uint8 [H, W]`` semantic labels.
        footprint: ``uint8 [H, W]`` binary exterior envelope.
    """

    svg_path = Path(svg_path)
    output_shape = (
        (int(image_size), int(image_size))
        if output_shape is None
        else tuple(int(value) for value in output_shape)
    )
    if (
        len(output_shape) != 2
        or min(output_shape) <= 0
    ):
        raise ValueError("output_shape must contain two positive dimensions")
    output_h, output_w = output_shape
    root = ET.parse(svg_path).getroot()
    view_box = _numbers(root.get("viewBox") or "")
    if len(view_box) != 4:
        width = float(root.get("width") or image_size)
        height = float(root.get("height") or image_size)
        min_x, min_y = 0.0, 0.0
    else:
        min_x, min_y, width, height = view_box
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"invalid SVG extent in {svg_path}: {width}x{height}")

    # Add a small margin so flood filling has an unambiguous outside seed.
    margin = max(3, min(output_h, output_w) // 64)
    scale = max(
        1.0,
        min(
            (output_w - 2 * margin) / width,
            (output_h - 2 * margin) / height,
        ),
    )
    render_w = max(1, round(width * scale))
    render_h = max(1, round(height * scale))
    canvas_w = render_w + 2 * margin
    canvas_h = render_h + 2 * margin
    normalize = np.array(
        [
            [scale, 0.0, margin - min_x * scale],
            [0.0, scale, margin - min_y * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    background = 0
    wall_id = taxonomy.class_id("wall") if taxonomy else 1
    structure_id = taxonomy.class_id("structure") if taxonomy else 12
    target_image = Image.new("L", (canvas_w, canvas_h), background)
    footprint_image = Image.new("L", (canvas_w, canvas_h), 0)
    target_draw = ImageDraw.Draw(target_image)
    footprint_draw = ImageDraw.Draw(footprint_image)

    def visit(element: ET.Element, parent_matrix: np.ndarray) -> None:
        matrix = parent_matrix @ parse_transform(element.get("transform"))
        element_id = element.get("id") or ""
        class_name = element.get("class") or ""
        tokens = class_name.split()

        is_space = "Space" in tokens
        is_outdoor = "Outdoor" in tokens
        if is_space and not is_outdoor:
            polygons = _direct_polygon_points(element, matrix)
            if polygons:
                if taxonomy and len(tokens) > 1:
                    target_name = taxonomy.room_map.get(tokens[1], "room_rare")
                    class_value = taxonomy.class_id(target_name)
                else:
                    class_value = room_class(class_name)
                _draw_polygons(target_draw, polygons, class_value, canvas_w, canvas_h)
                _draw_polygons(footprint_draw, polygons, 1, canvas_w, canvas_h)

        if element_id == "Wall":
            polygons = _direct_polygon_points(element, matrix)
            _draw_polygons(target_draw, polygons, wall_id, canvas_w, canvas_h)
        elif element_id in STRUCTURE_IDS:
            polygons = _direct_polygon_points(element, matrix)
            _draw_polygons(target_draw, polygons, structure_id, canvas_w, canvas_h)
        elif element_id == "Door":
            polygons = _direct_polygon_points(element, matrix)
            if taxonomy:
                subtype = " ".join(tokens[1:]) if len(tokens) > 1 else "Door"
                class_value = taxonomy.class_id(taxonomy.door_map.get(subtype, f"door:{subtype}"))
            else:
                class_value = 2
            _draw_polygons(target_draw, polygons, class_value, canvas_w, canvas_h)
        elif element_id == "Window":
            polygons = _direct_polygon_points(element, matrix)
            if taxonomy:
                subtype = " ".join(tokens[1:]) if len(tokens) > 1 else "Window"
                class_value = taxonomy.class_id(taxonomy.window_map.get(subtype, f"window:{subtype}"))
            else:
                class_value = 3
            _draw_polygons(target_draw, polygons, class_value, canvas_w, canvas_h)
        elif "FixedFurniture" in tokens and "FixedFurnitureSet" not in tokens:
            polygons = _furniture_boundary_points(element, matrix)
            if taxonomy and len(tokens) > 1:
                target_name = taxonomy.furniture_map.get(tokens[1], "furniture_rare")
                class_value = taxonomy.class_id(target_name)
            else:
                class_value = furniture_class(class_name)
            _draw_polygons(target_draw, polygons, class_value, canvas_w, canvas_h)

        for child in element:
            visit(child, matrix)

    floors = _floor_nodes(root, normalize)
    if floor_id is not None:
        requested = str(floor_id)
        if not requested.startswith("Floor-"):
            requested = f"Floor-{int(floor_id)}"
        floor_lookup = {item_id: parent for item_id, parent in floors}
        if requested not in floor_lookup:
            raise ValueError(f"floor {requested} not found in {svg_path}; available={sorted(floor_lookup)}")
        floor_node = next(node for node in root.iter() if (node.get("id") or "") == requested)
        visit(floor_node, floor_lookup[requested])
    elif floors:
        for available_id, parent in floors:
            floor_node = next(node for node in root.iter() if (node.get("id") or "") == available_id)
            visit(floor_node, parent)
    else:
        visit(root, normalize)

    target = np.array(target_image, dtype=np.uint8, copy=True)
    footprint = np.array(footprint_image, dtype=np.uint8, copy=True)
    if not footprint.any():
        footprint = (target > 0).astype(np.uint8)
    footprint = _fill_enclosed_holes(footprint)
    # CubiCasa room polygons stop at exterior wall faces. Grow the envelope just
    # enough to include the known exterior wall while excluding outdoor decks.
    dilation = max(3, 2 * math.ceil(4.0 * scale) + 1)
    footprint = np.array(
        Image.fromarray(footprint * 255, mode="L").filter(ImageFilter.MaxFilter(dilation)),
        dtype=np.uint8,
        copy=True,
    )
    footprint = (footprint > 0).astype(np.uint8)
    footprint = _fill_enclosed_holes(footprint)
    target[footprint == 0] = background

    pad_y = max(0, output_h - canvas_h)
    pad_x = max(0, output_w - canvas_w)
    top = pad_y // 2
    left = pad_x // 2
    target_out = np.zeros((output_h, output_w), dtype=np.uint8)
    footprint_out = np.zeros((output_h, output_w), dtype=np.uint8)
    copy_h = min(canvas_h, output_h)
    copy_w = min(canvas_w, output_w)
    target_out[top : top + copy_h, left : left + copy_w] = target[:copy_h, :copy_w]
    footprint_out[top : top + copy_h, left : left + copy_w] = footprint[:copy_h, :copy_w]
    return target_out, footprint_out

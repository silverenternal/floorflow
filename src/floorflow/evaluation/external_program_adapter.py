"""Convert CubiCasa room rasters into HouseGAN/HouseDiff oracle programs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from floorflow.data.taxonomy import FloorplanTaxonomy

HOUSE_ROOM_TYPES = {
    "room:LivingRoom": 1,
    "room:Kitchen": 2,
    "room:Bedroom": 3,
    "room:Bath": 4,
    "room:Sauna": 4,
    "room:Alcove": 5,
    "room:Den": 5,
    "room:Entry": 6,
    "room:Hall": 6,
    "room:DraughtLobby": 6,
    "room:Dining": 7,
    "room:Office": 8,
    "room:Closet": 10,
    "room:Storage": 10,
}


@dataclass(frozen=True)
class OracleRoom:
    class_name: str
    house_type: int
    box: tuple[float, float, float, float]
    area: int


def house_room_type(class_name: str) -> int:
    """Map a CubiCasa room class to the HouseGAN taxonomy."""
    return HOUSE_ROOM_TYPES.get(class_name, 16)


def extract_oracle_rooms(
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    min_area: int = 8,
    min_box_pixels: int = 3,
) -> list[OracleRoom]:
    """Recover one rectangle proposal per connected CubiCasa room component."""
    if target.ndim != 2:
        raise ValueError("target must be a two-dimensional label map")
    height, width = target.shape
    rooms: list[OracleRoom] = []
    for class_id in taxonomy.room_ids:
        class_name = taxonomy.class_names[int(class_id)]
        components, count = ndimage.label(target == int(class_id))
        for component_id in range(1, int(count) + 1):
            rows, columns = np.where(components == component_id)
            area = int(rows.size)
            if area < int(min_area):
                continue
            x0 = int(columns.min())
            y0 = int(rows.min())
            x1 = int(columns.max()) + 1
            y1 = int(rows.max()) + 1
            minimum = max(1, int(min_box_pixels))
            if x1 - x0 < minimum:
                extra = minimum - (x1 - x0)
                x0 = max(0, x0 - extra // 2)
                x1 = min(width, x0 + minimum)
                x0 = max(0, x1 - minimum)
            if y1 - y0 < minimum:
                extra = minimum - (y1 - y0)
                y0 = max(0, y0 - extra // 2)
                y1 = min(height, y0 + minimum)
                y0 = max(0, y1 - minimum)
            box = (
                float(x0) / width,
                float(y0) / height,
                float(x1) / width,
                float(y1) / height,
            )
            rooms.append(
                OracleRoom(
                    class_name=class_name,
                    house_type=house_room_type(class_name),
                    box=box,
                    area=area,
                )
            )
    rooms.sort(key=lambda room: (-room.area, room.box))
    return rooms


def _touching(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return max(dx, dy) <= float(tolerance)


def plan_to_housegan_payload(
    rooms: list[OracleRoom],
    *,
    tolerance: float = 0.03,
    resolution: int = 256,
) -> dict:
    """Build the JSON contract consumed by both HouseGAN++ and HouseDiffusion."""
    if not rooms:
        raise ValueError("oracle program requires at least one room")
    scale = float(resolution)
    boxes = [
        [float(round(coordinate * scale)) for coordinate in room.box]
        for room in rooms
    ]
    room_types = [int(room.house_type) for room in rooms]
    edges: list[list[float | int]] = []
    edge_rooms: list[list[int]] = []
    corners = (
        ((0, 1), (2, 1)),
        ((2, 1), (2, 3)),
        ((2, 3), (0, 3)),
        ((0, 3), (0, 1)),
    )
    for index, (box, room_type) in enumerate(zip(boxes, room_types)):
        for start, end in corners:
            edges.append(
                [
                    float(box[start[0]]),
                    float(box[start[1]]),
                    float(box[end[0]]),
                    float(box[end[1]]),
                    int(room_type),
                    int(index),
                ]
            )
            edge_rooms.append([int(index)])
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            if not _touching(rooms[first].box, rooms[second].box, tolerance):
                continue
            first_edge = 4 * first
            second_edge = 4 * second
            if second not in edge_rooms[first_edge]:
                edge_rooms[first_edge].append(second)
            if first not in edge_rooms[second_edge]:
                edge_rooms[second_edge].append(first)
    return {
        "room_type": room_types,
        "boxes": boxes,
        "edges": edges,
        "ed_rm": edge_rooms,
        "oracle_room_classes": [room.class_name for room in rooms],
    }


def write_housegan_oracle_dataset(
    dataset,
    output_dir: str | Path,
    *,
    min_area: int = 8,
    tolerance: float = 0.03,
    list_name: str = "sample_list.txt",
    list_relative: bool = False,
    list_directory: str | Path | None = None,
) -> dict:
    """Write one JSON per CubiCasa floor and a HouseGAN-compatible list."""
    output_dir = Path(output_dir)
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    records = []
    paths = []
    for index in range(len(dataset)):
        target, _footprint, _floor_id, name = dataset[index]
        rooms = extract_oracle_rooms(
            target.cpu().numpy() if hasattr(target, "cpu") else np.asarray(target),
            dataset.taxonomy,
            min_area=min_area,
        )
        if not rooms:
            continue
        payload = plan_to_housegan_payload(rooms, tolerance=tolerance)
        stem = str(name).replace("/", "__").replace(" ", "_")
        path = json_dir / f"{index:05d}_{stem}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        paths.append(path.resolve())
        records.append(
            {
                "id": str(name),
                "json": str(path),
                "rooms": len(rooms),
                "oracle_program": True,
            }
        )
    list_path = Path(list_directory or output_dir) / list_name
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(
        "".join(
            f"{path.name if list_relative else path}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    manifest = {
        "format": "housegan_rplan_json",
        "cases": len(records),
        "oracle_program": True,
        "sample_list": str(list_path),
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "OracleRoom",
    "extract_oracle_rooms",
    "house_room_type",
    "plan_to_housegan_payload",
    "write_housegan_oracle_dataset",
]

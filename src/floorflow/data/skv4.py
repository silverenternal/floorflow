"""Rasterization helpers for the skv4 plan-level test format."""

from __future__ import annotations

import cv2
import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy


def skv4_room_name(label: str) -> str:
    compact = str(label).replace("_", "").replace(" ", "")
    aliases = {
        "LIVING": "LivingRoom",
        "LIVINGROOM": "LivingRoom",
        "BEDROOM": "Bedroom",
        "BATHROOM": "Bath",
        "WC": "Bath",
        "KITCHEN": "Kitchen",
        "DINING": "Dining",
        "ENTRY": "Entry",
        "FOYER": "Entry",
        "HALL": "Hall",
        "CLOSET": "Closet",
        "WIC": "Closet",
        "STORAGE": "Storage",
        "STORE": "Storage",
        "UTILITY": "Utility",
        "TECHNICAL": "TechnicalRoom",
        "OFFICE": "Office",
        "GARAGE": "Garage",
        "SAUNA": "Sauna",
        "CORRIDOR": "Hall",
        "BALCONY": "Room",
    }
    return aliases.get(compact.upper(), label.title().replace(" ", ""))


def rasterize_skv4_plan(
    plan: dict,
    taxonomy: FloorplanTaxonomy,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = float(plan.get("frame", 1000.0))
    scale = image_size / frame
    target = np.zeros((image_size, image_size), dtype=np.uint8)
    footprint = np.zeros((image_size, image_size), dtype=np.uint8)
    for room in plan["gt"]["rooms"]:
        polygon = np.asarray(room.get("polygon", []), dtype=np.float64) * scale
        if len(polygon) < 3:
            continue
        points = np.round(polygon).astype(np.int32)
        name = skv4_room_name(room.get("label", "Undefined"))
        class_id = taxonomy.room_map.get(name, taxonomy.room_map.get("Undefined"))
        if class_id is None:
            continue
        if isinstance(class_id, str):
            class_id = taxonomy.class_id(class_id)
        cv2.fillPoly(target, [points], int(class_id))
        cv2.fillPoly(footprint, [points], 1)
    wall_id = taxonomy.class_id("wall")
    door_id = taxonomy.opening_ids[0]
    window_id = next(
        (
            class_id
            for class_id in taxonomy.opening_ids
            if taxonomy.class_names[class_id].startswith("window:")
        ),
        door_id,
    )
    for wall in plan["gt"]["walls"]:
        start = np.asarray(wall["start"], dtype=np.float64) * scale
        end = np.asarray(wall["end"], dtype=np.float64) * scale
        thickness = max(1, int(round(float(wall.get("thickness", 4.0)) * scale)))
        points = np.round(np.stack([start, end])).astype(np.int32)
        cv2.line(target, tuple(points[0]), tuple(points[1]), int(wall_id), thickness)
        cv2.line(footprint, tuple(points[0]), tuple(points[1]), 1, thickness)
        for opening in wall.get("openings", []):
            center = np.asarray(opening["center"], dtype=np.float64) * scale
            width = max(1, int(round(float(opening.get("width", 2.0)) * scale)))
            direction = end - start
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-6:
                continue
            offset = direction / norm * (width / 2.0)
            left = np.round(center - offset).astype(np.int32)
            right = np.round(center + offset).astype(np.int32)
            class_id = window_id if opening.get("type") == "window" else door_id
            cv2.line(target, tuple(left), tuple(right), int(class_id), max(2, thickness))
    footprint = np.maximum(footprint, (target > 0).astype(np.uint8))
    return target, footprint


__all__ = ["rasterize_skv4_plan", "skv4_room_name"]

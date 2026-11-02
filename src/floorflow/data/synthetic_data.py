"""Convert fpvec-lab fpgen scenes into floorflow raster caches."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from floorflow.data.taxonomy import FloorplanTaxonomy


ROOM_ALIASES = {
    "bedroom": "room:Bedroom",
    "bath": "room:Bath",
    "wc": "room:Bath",
    "powder": "room:Bath",
    "kitchen": "room:Kitchen",
    "living": "room:LivingRoom",
    "dining": "room:LivingRoom",
    "study": "room:Office",
    "store": "room:Storage",
    "storage": "room:Storage",
    "wardrobe": "room:Storage",
    "wic": "room:Storage",
    "foyer": "room:Entry",
    "entrance": "room:Entry",
    "lobby": "room:Entry",
    "hallway": "room:Hall",
    "corridor": "room:Hall",
    "yard": "room:Utility",
    "utility": "room:Utility",
    "shelter": "room:TechnicalRoom",
    "balcony": "room_rare",
    "pes": "room_rare",
    "ledge": "room_rare",
    "void": "room_rare",
    "planter": "room_rare",
}

FIXTURE_ALIASES = {
    "toilet": "furniture:Toilet",
    "basin": "furniture:Sink",
    "sink": "furniture:Sink",
    "bathtub": "furniture:Bathtub",
    "shower": "furniture:Shower",
    "counter": "furniture:BaseCabinet",
    "wardrobe": "furniture:Closet",
    "shoe_cabinet": "furniture:CoatCloset",
    "shelving": "furniture:WallCabinet",
    "bookshelf": "furniture:WallCabinet",
    "stove": "furniture:ElectricalAppliance",
    "fridge": "furniture:ElectricalAppliance",
    "washer": "furniture:ElectricalAppliance",
    "dryer": "furniture:ElectricalAppliance",
    "ac_unit": "furniture:ElectricalAppliance",
}

DOOR_CLASS = "door:Swing Beside"
WINDOW_CLASS = "window:Regular"


def _room_class(label: str, taxonomy: FloorplanTaxonomy) -> int:
    normalized = label.upper()
    if not normalized:
        return taxonomy.class_id("room:Hall")
    for alias, target in ROOM_ALIASES.items():
        if alias.upper() in normalized:
            return taxonomy.class_id(target)
    return taxonomy.class_id("room_rare")


def _draw_polygon(draw: ImageDraw.ImageDraw, polygon: list, value: int) -> None:
    if len(polygon) >= 3:
        draw.polygon([(float(x), float(y)) for x, y in polygon], fill=int(value))


def _draw_wall(
    draw: ImageDraw.ImageDraw,
    start: list[float],
    end: list[float],
    thickness: float,
    value: int,
) -> None:
    width = max(1, int(round(float(thickness))))
    draw.line(
        [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))],
        fill=int(value),
        width=width,
    )
    radius = width / 2.0
    for x, y in (start, end):
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=int(value),
        )


def convert_fpgen_gt(
    gt_path: str | Path,
    taxonomy: FloorplanTaxonomy,
    *,
    image_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    meta = payload.get("meta", {})
    image_width = float(meta.get("img_w", 1024))
    image_height = float(meta.get("img_h", 1024))
    canvas_width = 1024
    canvas_height = max(1, int(round(canvas_width * image_height / max(1.0, image_width))))
    target_image = Image.new("L", (canvas_width, canvas_height), 0)
    footprint_image = Image.new("L", (canvas_width, canvas_height), 0)
    target_draw = ImageDraw.Draw(target_image)
    footprint_draw = ImageDraw.Draw(footprint_image)
    for room in payload.get("rooms", []):
        value = _room_class(room.get("label", ""), taxonomy)
        _draw_polygon(target_draw, room.get("polygon", []), value)
        _draw_polygon(footprint_draw, room.get("polygon", []), 1)
    wall_id = taxonomy.class_id("wall")
    door_id = taxonomy.class_id(DOOR_CLASS)
    window_id = taxonomy.class_id(WINDOW_CLASS)
    for wall in payload.get("walls", []):
        _draw_wall(
            target_draw,
            wall["start"],
            wall["end"],
            wall.get("thickness", 3),
            wall_id,
        )
        _draw_wall(
            footprint_draw,
            wall["start"],
            wall["end"],
            wall.get("thickness", 3),
            1,
        )
        start = np.asarray(wall["start"], dtype=np.float64)
        end = np.asarray(wall["end"], dtype=np.float64)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            continue
        direction /= length
        for opening in wall.get("openings", []):
            center = np.asarray(opening["center"], dtype=np.float64)
            half = float(opening["width"]) / 2.0
            p0 = center - direction * half
            p1 = center + direction * half
            value = door_id if opening.get("type") != "window" else window_id
            _draw_wall(
                target_draw,
                p0.tolist(),
                p1.tolist(),
                wall.get("thickness", 3),
                value,
            )
    for fixture in payload.get("fixtures", []):
        target = FIXTURE_ALIASES.get(str(fixture.get("kind", "")).lower())
        if target is not None and taxonomy.class_id(target) > 0:
            _draw_polygon(
                target_draw,
                fixture.get("quad", []),
                taxonomy.class_id(target),
            )
    footprint = np.asarray(footprint_image, dtype=np.uint8)
    footprint = np.asarray(
        Image.fromarray(footprint * 255).filter(ImageFilter.MaxFilter(5)),
        dtype=np.uint8,
    )
    footprint = (footprint > 127).astype(np.uint8)
    target = np.asarray(target_image, dtype=np.uint8)
    scale = min(image_size / canvas_width, image_size / canvas_height)
    resized_w = max(1, int(round(canvas_width * scale)))
    resized_h = max(1, int(round(canvas_height * scale)))
    target = np.asarray(
        Image.fromarray(target).resize((resized_w, resized_h), Image.Resampling.NEAREST)
    )
    footprint = np.asarray(
        Image.fromarray(footprint).resize((resized_w, resized_h), Image.Resampling.NEAREST)
    )
    target_out = np.zeros((image_size, image_size), dtype=np.uint8)
    footprint_out = np.zeros((image_size, image_size), dtype=np.uint8)
    top = (image_size - resized_h) // 2
    left = (image_size - resized_w) // 2
    target_out[top : top + resized_h, left : left + resized_w] = target
    footprint_out[top : top + resized_h, left : left + resized_w] = footprint
    target_out[footprint_out == 0] = 0
    return target_out, footprint_out


class SyntheticFloorplanDataset:
    """Cached synthetic plans with the CubiCasa dataset interface."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        *,
        augment: bool = False,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.split = split
        self.augment = bool(augment)
        self.taxonomy = FloorplanTaxonomy.read(self.cache_root / "taxonomy.json")
        payload = json.loads((self.cache_root / "manifest.json").read_text(encoding="utf-8"))
        self.samples = [self.cache_root / relative for relative in payload["splits"][split]]
        if not self.samples:
            raise FileNotFoundError(f"synthetic split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        path = self.samples[index]
        with np.load(path) as payload:
            target = payload["target"].astype(np.uint8, copy=True)
            footprint = payload["footprint"].astype(np.uint8, copy=True)
        if self.augment:
            if random.random() < 0.5:
                target = np.flip(target, axis=1)
                footprint = np.flip(footprint, axis=1)
            if random.random() < 0.5:
                target = np.flip(target, axis=0)
                footprint = np.flip(footprint, axis=0)
            rotations = random.randrange(4)
            if rotations:
                target = np.rot90(target, k=rotations)
                footprint = np.rot90(footprint, k=rotations)
        return (
            torch.from_numpy(np.ascontiguousarray(target)).long(),
            torch.from_numpy(np.ascontiguousarray(footprint)).float(),
            1,
            path.stem,
        )

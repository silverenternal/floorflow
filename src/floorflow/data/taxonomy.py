"""CubiCasa5K source taxonomy with data-driven rare-label grouping."""

from __future__ import annotations

import colorsys
import json
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

BACKGROUND = "background"
WALL = "wall"
STRUCTURE = "structure"
ROOM_RARE = "room_rare"
FURNITURE_RARE = "furniture_rare"

STRUCTURE_IDS = {"Railing", "Column", "Stairs", "StairWell", "Elevator"}


def _tokens(class_name: str) -> list[str]:
    return [token for token in class_name.split() if token]


def scan_taxonomy_counts(svg_paths: list[Path]) -> dict[str, Counter[str]]:
    counts = {
        "room": Counter(),
        "furniture": Counter(),
        "door": Counter(),
        "window": Counter(),
    }
    for path in svg_paths:
        root = ET.parse(path).getroot()
        for element in root.iter():
            tokens = _tokens(element.get("class") or "")
            element_id = element.get("id") or ""
            if "Space" in tokens and len(tokens) > 1 and tokens[1] != "Outdoor":
                counts["room"][tokens[1]] += 1
            if "FixedFurniture" in tokens and len(tokens) > 1 and "FixedFurnitureSet" not in tokens:
                counts["furniture"][tokens[1]] += 1
            if element_id == "Door":
                subtype = " ".join(tokens[1:]) if len(tokens) > 1 else "Door"
                counts["door"][subtype] += 1
            if element_id == "Window":
                subtype = " ".join(tokens[1:]) if len(tokens) > 1 else "Window"
                counts["window"][subtype] += 1
    return counts


def _color(index: int) -> tuple[int, int, int]:
    if index == 0:
        return (255, 255, 255)
    hue = (index * 0.61803398875) % 1.0
    saturation = 0.35 + 0.35 * ((index * 7) % 3) / 2.0
    value = 0.95
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(red * 255), int(green * 255), int(blue * 255)


@dataclass(frozen=True)
class FloorplanTaxonomy:
    class_names: tuple[str, ...]
    room_map: dict[str, str]
    furniture_map: dict[str, str]
    door_map: dict[str, str]
    window_map: dict[str, str]
    source_counts: dict[str, dict[str, int]]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_target_channels(self) -> int:
        return self.num_classes - 1

    @property
    def class_to_id(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}

    @property
    def room_ids(self) -> tuple[int, ...]:
        return tuple(index for index, name in enumerate(self.class_names) if self.is_room(name))

    @property
    def furniture_ids(self) -> tuple[int, ...]:
        return tuple(index for index, name in enumerate(self.class_names) if self.is_furniture(name))

    @property
    def opening_ids(self) -> tuple[int, ...]:
        return tuple(index for index, name in enumerate(self.class_names) if self.is_opening(name))

    def is_room(self, name: str) -> bool:
        return name.startswith("room:") or name == ROOM_RARE

    def is_furniture(self, name: str) -> bool:
        return name.startswith("furniture:") or name == FURNITURE_RARE

    def is_opening(self, name: str) -> bool:
        return name.startswith("door:") or name.startswith("window:")

    def class_id(self, name: str) -> int:
        try:
            return self.class_names.index(name)
        except ValueError:
            return 0

    def color(self, class_id: int) -> tuple[int, int, int]:
        return _color(int(class_id))

    def channel_prior(self) -> tuple[tuple[float, ...], ...]:
        size = self.num_target_channels
        matrix = [[0.0] * size for _ in range(size)]

        def link(left: int, right: int, weight: float) -> None:
            if left <= 0 or right <= 0:
                return
            matrix[left - 1][right - 1] = weight
            matrix[right - 1][left - 1] = weight

        wall = self.class_id("wall")
        structure = self.class_id("structure")
        for class_id, name in enumerate(self.class_names):
            if name.startswith("door:") or name.startswith("window:"):
                link(class_id, wall, 0.9)
            if self.is_room(name) and name in {"room:Bath", "room:Sauna"}:
                for furniture_id in self.furniture_ids:
                    link(class_id, furniture_id, 0.5)
            if name == "room:Kitchen":
                for furniture_name in (
                    "furniture:Sink",
                    "furniture:ElectricalAppliance",
                    "furniture:BaseCabinet",
                    "furniture:WallCabinet",
                ):
                    link(class_id, self.class_id(furniture_name), 0.8)
            if name in {"room:LivingRoom", "room:Den"}:
                for furniture_name in ("furniture:Fireplace", "furniture:FireplaceCorner"):
                    link(class_id, self.class_id(furniture_name), 0.8)
            if name in {"room:Closet", "room:Storage", "room:DressingRoom"}:
                for furniture_name in (
                    "furniture:Closet",
                    "furniture:CoatCloset",
                    "furniture:WallCabinet",
                ):
                    link(class_id, self.class_id(furniture_name), 0.7)
            if name == "structure":
                link(class_id, wall, 0.6)
        return tuple(tuple(row) for row in matrix)

    def to_json(self) -> dict:
        return {
            "class_names": self.class_names,
            "room_map": self.room_map,
            "furniture_map": self.furniture_map,
            "door_map": self.door_map,
            "window_map": self.window_map,
            "source_counts": self.source_counts,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "FloorplanTaxonomy":
        return cls(
            class_names=tuple(payload["class_names"]),
            room_map=dict(payload["room_map"]),
            furniture_map=dict(payload["furniture_map"]),
            door_map=dict(payload["door_map"]),
            window_map=dict(payload["window_map"]),
            source_counts=dict(payload["source_counts"]),
        )

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "FloorplanTaxonomy":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def build_taxonomy(
    counts: dict[str, Counter[str]],
    *,
    room_min_count: int = 100,
    furniture_min_count: int = 100,
) -> FloorplanTaxonomy:
    names = [BACKGROUND, WALL]

    for subtype in sorted(counts.get("door", {})):
        names.append(f"door:{subtype}")
    for subtype in sorted(counts.get("window", {})):
        names.append(f"window:{subtype}")

    room_map: dict[str, str] = {}
    for subtype, count in sorted(counts.get("room", {}).items()):
        target = f"room:{subtype}" if count >= room_min_count else ROOM_RARE
        room_map[subtype] = target
        if target not in names:
            names.append(target)

    furniture_map: dict[str, str] = {}
    for subtype, count in sorted(counts.get("furniture", {}).items()):
        target = f"furniture:{subtype}" if count >= furniture_min_count else FURNITURE_RARE
        furniture_map[subtype] = target
        if target not in names:
            names.append(target)

    names.append(STRUCTURE)
    return FloorplanTaxonomy(
        class_names=tuple(names),
        room_map=room_map,
        furniture_map=furniture_map,
        door_map={subtype: f"door:{subtype}" for subtype in counts.get("door", {})},
        window_map={subtype: f"window:{subtype}" for subtype in counts.get("window", {})},
        source_counts={key: dict(value) for key, value in counts.items()},
    )

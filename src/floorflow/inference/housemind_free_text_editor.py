"""Deterministic free-text editor for HouseMind-style color floorplans."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label

RGB = tuple[int, int, int]
ROOM_COLORS: dict[str, RGB] = {
    "LivingRoom": (238, 232, 170),
    "MasterRoom": (255, 165, 0),
    "Kitchen": (240, 128, 128),
    "Bathroom": (173, 216, 210),
    "Balcony": (107, 142, 35),
    "DiningRoom": (218, 112, 214),
    "Storage": (221, 160, 221),
    "CommonRoom": (255, 215, 0),
}
ROOM_ALIASES = {
    "LivingRoom": ("living room", "livingroom", "客厅"),
    "MasterRoom": ("master room", "masterroom", "主卧", "主卧室"),
    "Kitchen": ("kitchen", "厨房"),
    "Bathroom": ("bathroom", "toilet", "卫生间", "浴室", "洗手间"),
    "Balcony": ("balcony", "阳台"),
    "DiningRoom": ("dining room", "diningroom", "餐厅"),
    "Storage": ("storage", "储藏", "储物"),
    "CommonRoom": (
        "common room",
        "commonroom",
        "second room",
        "secondroom",
        "study room",
        "studyroom",
        "child room",
        "childroom",
        "guest room",
        "guestroom",
        "bedroom",
        "次卧",
        "书房",
        "儿童房",
        "客卧",
        "卧室",
    ),
}
ADD_WORDS = (
    "add",
    "create",
    "include",
    "insert",
    "place",
    "put",
    "设置",
    "布置",
    "新增",
    "增加",
    "添加",
    "创建",
    "加在",
    "加入",
)
DELETE_WORDS = (
    "delete",
    "remove",
    "eliminate",
    "erase",
    "删除",
    "移除",
    "去掉",
    "去除",
    "清除",
)
DIRECTIONS = {
    "northwest": (0.15, 0.15),
    "northeast": (0.85, 0.15),
    "southwest": (0.15, 0.85),
    "southeast": (0.85, 0.85),
    "north": (0.5, 0.15),
    "south": (0.5, 0.85),
    "west": (0.15, 0.5),
    "east": (0.85, 0.5),
    "center": (0.5, 0.5),
    "西北": (0.15, 0.15),
    "东北": (0.85, 0.15),
    "西南": (0.15, 0.85),
    "东南": (0.85, 0.85),
    "北": (0.5, 0.15),
    "南": (0.5, 0.85),
    "西": (0.15, 0.5),
    "东": (0.85, 0.5),
    "中央": (0.5, 0.5),
}
BASE_AREA_FRACTION = {
    "Balcony": 0.065,
    "Bathroom": 0.045,
    "Kitchen": 0.07,
    "MasterRoom": 0.1,
    "CommonRoom": 0.08,
    "DiningRoom": 0.075,
    "Storage": 0.04,
    "LivingRoom": 0.12,
}
DEFAULT_ADD_DIRECTION = {
    "Balcony": "south",
    "Bathroom": "east",
    "Kitchen": "north",
    "MasterRoom": "east",
    "CommonRoom": "east",
    "DiningRoom": "center",
    "Storage": "west",
    "LivingRoom": "center",
}


@dataclass(frozen=True)
class EditRequest:
    operation: str
    room_type: str
    direction: str | None
    size: str | None


def parse_instruction(text: str) -> EditRequest | None:
    lower = re.sub(r"\s+", " ", text.lower())
    operation = (
        "add"
        if any(word in lower for word in ADD_WORDS)
        else "delete"
        if any(word in lower for word in DELETE_WORDS)
        else None
    )
    if operation is None:
        return None
    room_type = next(
        (
            canonical
            for canonical, aliases in ROOM_ALIASES.items()
            if any(alias in lower for alias in aliases)
        ),
        None,
    )
    if room_type is None:
        return None
    direction = next(
        (
            canonical
            for alias, canonical in (
                ("西北", "northwest"),
                ("东北", "northeast"),
                ("西南", "southwest"),
                ("东南", "southeast"),
                ("northwest", "northwest"),
                ("northeast", "northeast"),
                ("southwest", "southwest"),
                ("southeast", "southeast"),
                ("北", "north"),
                ("南", "south"),
                ("东", "east"),
                ("西", "west"),
                ("north", "north"),
                ("south", "south"),
                ("west", "west"),
                ("east", "east"),
                ("center", "center"),
                ("central", "center"),
            )
            if alias in lower
        ),
        None,
    )
    size = (
        "small"
        if any(word in lower for word in ("small", "小"))
        else "large"
        if any(word in lower for word in ("large", "big", "大"))
        else "medium"
        if any(word in lower for word in ("medium", "中型"))
        else None
    )
    return EditRequest(operation, room_type, direction, size)


def palette_labels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colors = np.asarray(
        [ROOM_COLORS[name] for name in ROOM_COLORS] + [(255, 255, 255), (0, 0, 0)],
        dtype=np.int16,
    )
    flat = image.astype(np.int32).reshape(-1, 3)
    indices = np.argmin(
        np.sum((flat[:, None, :] - colors[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    return indices.reshape(image.shape[:2]), colors.astype(np.uint8)


def components(mask: np.ndarray) -> list[np.ndarray]:
    labeled, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return [labeled == index for index in range(1, count + 1)]


def direction_distance(
    component: np.ndarray,
    direction: str | None,
    bounds: tuple[int, int, int, int],
) -> float:
    if direction is None:
        return 0.0
    top, bottom, left, right = bounds
    ys, xs = np.nonzero(component)
    x = (xs.mean() - left) / max(right - left, 1)
    y = (ys.mean() - top) / max(bottom - top, 1)
    target_x, target_y = DIRECTIONS[direction]
    return float(np.hypot(x - target_x, y - target_y))


def choose_component(
    image_labels: np.ndarray,
    room_type: str,
    direction: str | None,
) -> np.ndarray | None:
    color_index = list(ROOM_COLORS).index(room_type)
    rows = np.argwhere(image_labels < len(ROOM_COLORS))
    if not len(rows):
        return None
    bounds = (
        int(rows[:, 0].min()),
        int(rows[:, 0].max()),
        int(rows[:, 1].min()),
        int(rows[:, 1].max()),
    )
    candidates = components(image_labels == color_index)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            direction_distance(item, direction, bounds),
            int(item.sum()) if direction is None else -int(item.sum()),
        ),
    )


def edit_layout(
    image: np.ndarray,
    request: EditRequest,
) -> tuple[np.ndarray, dict[str, object]]:
    labels, colors = palette_labels(image)
    room_upper = len(ROOM_COLORS)
    output = image.astype(np.uint8).copy()
    target_index = list(ROOM_COLORS).index(request.room_type)
    target_color = np.asarray(ROOM_COLORS[request.room_type], dtype=np.uint8)
    colored = labels < room_upper
    if request.operation == "delete":
        selected = choose_component(labels, request.room_type, request.direction)
        if selected is None:
            return output, {"applied": False, "reason": "target_not_found"}
        white = labels == room_upper
        changed = selected | (
            white & binary_dilation(selected, iterations=2)
        )
        ring = binary_dilation(selected, iterations=2) & ~selected
        neighbors = labels[ring & colored & (labels != target_index)]
        if len(neighbors):
            values, counts = np.unique(neighbors, return_counts=True)
            donor_index = int(values[np.argmax(counts)])
        else:
            donor_index = room_upper
        output[changed] = (
            colors[donor_index]
            if donor_index < room_upper
            else np.asarray((255, 255, 255), dtype=np.uint8)
        )
        return output, {
            "applied": True,
            "target_pixels": int(selected.sum()),
            "changed_pixels": int(changed.sum()),
        }

    rows = np.argwhere(colored)
    if not len(rows):
        return output, {"applied": False, "reason": "empty_plan"}
    bounds = (
        int(rows[:, 0].min()),
        int(rows[:, 0].max()),
        int(rows[:, 1].min()),
        int(rows[:, 1].max()),
    )
    donor_candidates = [
        (name, components(labels == index))
        for index, name in enumerate(ROOM_COLORS)
        if name != request.room_type
    ]
    donor_components = [
        (name, item)
        for name, rows_for_name in donor_candidates
        for item in rows_for_name
        if item.sum() >= 16
    ]
    if not donor_components:
        return output, {"applied": False, "reason": "donor_not_found"}
    effective_direction = request.direction or DEFAULT_ADD_DIRECTION[
        request.room_type
    ]
    donor_type, donor = min(
        donor_components,
        key=lambda item: (
            direction_distance(item[1], effective_direction, bounds)
            if request.direction is not None
            else 0.0,
            -int(item[1].sum()),
        ),
    )
    fraction = BASE_AREA_FRACTION[request.room_type]
    if request.size == "small":
        fraction *= 0.65
    elif request.size == "large":
        fraction *= 1.5
    target_pixels = max(
        12,
        min(
            int(donor.sum() * 0.45),
            int(colored.sum() * fraction),
        ),
    )
    ys, xs = np.nonzero(donor)
    top, bottom, left, right = bounds
    if effective_direction in {"west", "east"}:
        score = xs - left if effective_direction == "west" else right - xs
    elif effective_direction in {"north", "south"}:
        score = ys - top if effective_direction == "north" else bottom - ys
    elif effective_direction == "northwest":
        score = (xs - left) + (ys - top)
    elif effective_direction == "northeast":
        score = (right - xs) + (ys - top)
    elif effective_direction == "southwest":
        score = (xs - left) + (bottom - ys)
    elif effective_direction == "southeast":
        score = (right - xs) + (bottom - ys)
    else:
        score = (xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2
    selected = np.zeros_like(donor)
    selected[ys[np.argsort(score)[:target_pixels]], xs[np.argsort(score)[:target_pixels]]] = True
    output[selected] = target_color
    separator = binary_dilation(selected, iterations=1) & donor & ~selected
    output[separator] = np.asarray((255, 255, 255), dtype=np.uint8)
    return output, {
        "applied": True,
        "donor": donor_type,
        "target_pixels": int(selected.sum()),
        "changed_pixels": int((selected | separator).sum()),
        "target_index": target_index,
    }

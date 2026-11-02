"""Style-free metrics for externally generated floorplan raster outputs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HOUSE_ROOM_COLORS = {
    1: "#EE4D4D",
    2: "#C67C7B",
    3: "#FFD274",
    4: "#BEBEBE",
    5: "#BFE3E8",
    6: "#7BA779",
    7: "#E87A90",
    8: "#FF8C69",
    10: "#1F849B",
    15: "#727171",
    16: "#785A67",
    17: "#D3A2C7",
}

HOUSE_TYPE_TO_CUBICASA = {
    1: {"room:LivingRoom"},
    2: {"room:Kitchen"},
    3: {"room:Bedroom"},
    4: {"room:Bath", "room:Sauna"},
    5: {"room:Alcove", "room:Den"},
    6: {"room:Entry", "room:Hall", "room:DraughtLobby"},
    7: {"room:Dining"},
    8: {"room:Office"},
    10: {"room:Closet", "room:Storage"},
}


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _read_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (size[1], size[0]):
        image = image.resize((size[1], size[0]), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def _support(image: np.ndarray) -> np.ndarray:
    return image.min(axis=-1) < 250


def evaluate_housegan_cases(
    dataset,
    output_dir: str | Path,
    *,
    cases: int | None = None,
) -> dict:
    """Measure coverage, footprint compliance and mapped room-class IoU."""
    output_dir = Path(output_dir)
    limit = len(dataset) if cases is None else min(int(cases), len(dataset))
    rows = []
    class_ious: dict[str, list[float]] = {}
    class_ids = {
        name: int(taxonomy_id)
        for taxonomy_id, name in enumerate(dataset.taxonomy.class_names)
    }
    for index in range(limit):
        target, footprint, _floor_id, name = dataset[index]
        target = np.asarray(target.cpu() if hasattr(target, "cpu") else target)
        footprint = np.asarray(
            footprint.cpu() if hasattr(footprint, "cpu") else footprint
        ).astype(bool)
        image = _read_rgb(
            output_dir / f"fp_final_{index}.png",
            target.shape,
        )
        generated = _support(image)
        target_support = np.isin(target, dataset.taxonomy.room_ids)
        intersection = np.logical_and(generated, target_support).sum()
        union = np.logical_or(generated, target_support).sum()
        class_scores = {}
        for house_type, cubicasa_names in HOUSE_TYPE_TO_CUBICASA.items():
            color = np.asarray(_rgb(HOUSE_ROOM_COLORS[house_type]))
            generated_class = np.linalg.norm(
                image.astype(np.int16) - color,
                axis=-1,
            ) <= 3
            target_class = np.zeros_like(generated_class)
            for class_name in cubicasa_names:
                if class_name in class_ids:
                    target_class |= target == class_ids[class_name]
            class_union = np.logical_or(generated_class, target_class).sum()
            if class_union:
                iou = float(
                    np.logical_and(generated_class, target_class).sum()
                    / class_union
                )
                class_scores[str(house_type)] = iou
                class_ious.setdefault(str(house_type), []).append(iou)
        rows.append(
            {
                "id": str(name),
                "generated_support": int(generated.sum()),
                "target_support": int(target_support.sum()),
                "room_support_iou": float(intersection / max(1, union)),
                "footprint_coverage": float(
                    np.logical_and(generated, footprint).sum()
                    / max(1, footprint.sum())
                ),
                "outside_footprint_pixels": int(
                    np.logical_and(generated, ~footprint).sum()
                ),
                "class_iou": class_scores,
            }
        )
    return {
        "method": "House-GAN++",
        "condition_mode": "oracle_room_program",
        "target_derived_control": True,
        "best_of_n_used": False,
        "cases": len(rows),
        "summary": {
            "room_support_iou_mean": float(
                np.mean([row["room_support_iou"] for row in rows])
            ),
            "footprint_coverage_mean": float(
                np.mean([row["footprint_coverage"] for row in rows])
            ),
            "outside_footprint_pixels_mean": float(
                np.mean([row["outside_footprint_pixels"] for row in rows])
            ),
            "mapped_class_iou_mean": float(
                np.mean(
                    [
                        np.mean(values)
                        for values in class_ious.values()
                        if values
                    ]
                )
            ),
            "mapped_class_iou_by_type": {
                key: float(np.mean(values))
                for key, values in sorted(class_ious.items())
            },
        },
        "rows": rows,
    }


def evaluate_maskplan_cases(
    output_dir: str | Path,
    boundary_dir: str | Path,
    site_ids: np.ndarray,
) -> dict:
    """Measure footprint compliance for native MaskPLAN raster outputs."""
    output_dir = Path(output_dir)
    boundary_dir = Path(boundary_dir)
    rows = []
    for raw_site in np.asarray(site_ids).tolist():
        site_id = int(raw_site)
        generated_path = output_dir / f"{site_id}.png"
        boundary_path = boundary_dir / f"{site_id}.png"
        if not generated_path.is_file() or not boundary_path.is_file():
            continue
        generated_image = cv2.imread(
            str(generated_path),
            cv2.IMREAD_UNCHANGED,
        )
        boundary_image = cv2.imread(
            str(boundary_path),
            cv2.IMREAD_UNCHANGED,
        )
        if generated_image is None or boundary_image is None:
            continue
        generated_alpha = (
            generated_image[:, :, -1]
            if generated_image.ndim == 3
            else generated_image
        )
        generated = generated_alpha > 0
        boundary_alpha = (
            boundary_image[:, :, -1]
            if boundary_image.ndim == 3
            else boundary_image
        )
        contours, _ = cv2.findContours(
            (boundary_alpha > 20).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        footprint = np.zeros(boundary_alpha.shape, dtype=np.uint8)
        if contours:
            cv2.drawContours(footprint, contours, -1, 1, thickness=-1)
        footprint = footprint.astype(bool)
        rows.append(
            {
                "id": site_id,
                "generated_support": int(generated.sum()),
                "footprint_coverage": float(
                    np.logical_and(generated, footprint).sum()
                    / max(1, footprint.sum())
                ),
                "outside_footprint_pixels": int(
                    np.logical_and(generated, ~footprint).sum()
                ),
            }
        )
    return {
        "method": "MaskPLAN",
        "condition_mode": "boundary_only_native_rplan",
        "target_derived_control": False,
        "best_of_n_used": False,
        "cases": len(rows),
        "summary": {
            "footprint_coverage_mean": float(
                np.mean([row["footprint_coverage"] for row in rows])
            ),
            "outside_footprint_pixels_mean": float(
                np.mean(
                    [row["outside_footprint_pixels"] for row in rows]
                )
            ),
        },
        "rows": rows,
    }
__all__ = [
    "HOUSE_ROOM_COLORS",
    "HOUSE_TYPE_TO_CUBICASA",
    "evaluate_housegan_cases",
    "evaluate_maskplan_cases",
]

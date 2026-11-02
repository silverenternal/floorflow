#!/usr/bin/env python3
"""Convert FloorFlow room instances to Raster2Seq COCO supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from floorflow.data.dataset import CubiCasaRasterControlDataset, load_or_rasterize
from floorflow.data.raster_control import load_cubicasa_rgb_raster
from floorflow.representation.core import extract_room_instances

ROOT = Path(__file__).resolve().parents[1]

CATEGORY_ID = {
    "room:Outdoor": 0,
    "room:Kitchen": 1,
    "room:LivingRoom": 2,
    "room:Bedroom": 3,
    "room:Bath": 4,
    "room:Entry": 5,
    "room:Storage": 6,
    "room:Garage": 7,
    "room:Undefined": 8,
}

CATEGORY_NAMES = {
    0: "Outdoor",
    1: "Kitchen",
    2: "Living Room",
    3: "Bed Room",
    4: "Bath",
    5: "Entry",
    6: "Storage",
    7: "Garage",
    8: "Undefined",
    9: "Window",
    10: "Door",
    11: "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--floorflow-cache",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/cache/raster2seq_floorflow_coco_v1",
    )
    return parser.parse_args()


def _category(class_name: str) -> int:
    if class_name in CATEGORY_ID:
        return CATEGORY_ID[class_name]
    return CATEGORY_ID["room:Undefined"]


def _contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)[:, 0, :]
    if len(contour) < 3 or cv2.contourArea(contour) <= 0.0:
        return None
    return contour.astype(np.float32)


def _hash_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    dataset = CubiCasaRasterControlDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.floorflow_cache,
        augment=False,
        limit=int(args.offset) + int(args.limit),
        image_mode="native",
    )
    samples = dataset.samples[int(args.offset) : int(args.offset) + int(args.limit)]
    if len(samples) != int(args.limit):
        raise RuntimeError(
            f"expected {args.limit} {args.split} samples, found {len(samples)}"
        )
    image_dir = args.output_root / args.split / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    provenance_rows = []
    annotation_id = 0
    for split_index, sample in enumerate(samples):
        sample_key = f"{sample.relative_case}__{sample.floor_id}"
        safe_key = sample_key.replace("/", "__")
        image_path = image_dir / f"{safe_key}.png"
        image = load_cubicasa_rgb_raster(sample.raster_path, image_size=256)
        Image.fromarray(
            np.rint(image * 255.0).clip(0, 255).astype(np.uint8)
        ).save(image_path)
        target, _footprint = load_or_rasterize(
            sample,
            image_size=256,
            taxonomy=dataset.taxonomy,
            write_cache=False,
        )
        instances = sorted(
            extract_room_instances(target, dataset.taxonomy, min_area=8),
            key=lambda value: value.area,
            reverse=True,
        )
        room_annotations = []
        for instance in instances:
            polygon = _contour(instance.mask)
            if polygon is None:
                continue
            x, y, width, height = instance.bbox
            room_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": split_index,
                    "category_id": _category(
                        dataset.taxonomy.class_names[int(instance.class_id)]
                    ),
                    "segmentation": [polygon.reshape(-1).tolist()],
                    "area": float(abs(cv2.contourArea(polygon))),
                    "bbox": [
                        float(x),
                        float(y),
                        float(max(1, width)),
                        float(max(1, height)),
                    ],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        if not room_annotations:
            image_path.unlink()
            continue
        images.append(
            {
                "id": split_index,
                "file_name": f"images/{safe_key}.png",
                "width": 256,
                "height": 256,
            }
        )
        annotations.extend(room_annotations)
        provenance_rows.append(
            {
                "sample_key": sample_key,
                "split_index": split_index,
                "source_raster": str(sample.raster_path),
                "image": str(image_path),
                "rooms": len(room_annotations),
            }
        )
    annotation_dir = args.output_root / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id, "name": name}
            for class_id, name in CATEGORY_NAMES.items()
        ],
    }
    annotation_path = annotation_dir / f"{args.split}.json"
    annotation_path.write_text(
        json.dumps(coco, indent=2) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "raster2seq-floorflow-coco-v1",
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "cases": len(images),
        "annotations": len(annotations),
        "target_used_at_inference": False,
        "target_role": "room-polygon supervision only",
        "taxonomy": dataset.taxonomy.to_json(),
        "taxonomy_hash": _hash_json(dataset.taxonomy.to_json()),
        "annotation_path": str(annotation_path),
        "rows": provenance_rows,
    }
    (args.output_root / f"{args.split}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "split": args.split,
                "cases": len(images),
                "annotations": len(annotations),
                "annotation_path": str(annotation_path),
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()

"""Create one deterministic geometric TTA variant of Raster2Seq val data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMS = {
    "hflip": Image.Transpose.FLIP_LEFT_RIGHT,
    "vflip": Image.Transpose.FLIP_TOP_BOTTOM,
    "rot180": Image.Transpose.ROTATE_180,
    "rot90": Image.Transpose.ROTATE_90,
    "rot270": Image.Transpose.ROTATE_270,
    "transpose": Image.Transpose.TRANSPOSE,
    "transverse": Image.Transpose.TRANSVERSE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs/cache/raster2seq_cc5k_val_refined_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--transform", choices=TRANSFORMS, required=True)
    return parser.parse_args()


def transform_polygon(
    flat: list[float],
    width: int,
    height: int,
    transform: str,
) -> list[float]:
    output = []
    for index in range(0, len(flat), 2):
        x, y = float(flat[index]), float(flat[index + 1])
        if transform in {"hflip", "rot180"}:
            x = width - 1.0 - x
        if transform in {"vflip", "rot180"}:
            y = height - 1.0 - y
        if transform == "rot90":
            x, y = y, width - 1.0 - x
        elif transform == "rot270":
            x, y = height - 1.0 - y, x
        elif transform == "transpose":
            x, y = y, x
        elif transform == "transverse":
            x, y = height - 1.0 - y, width - 1.0 - x
        output.extend([x, y])
    return output


def transform_bbox(
    polygon: list[float],
) -> list[float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def main() -> None:
    args = parse_args()
    source_annotation = args.source / "annotations" / f"{args.split}.json"
    payload = json.loads(source_annotation.read_text(encoding="utf-8"))
    image_rows = {int(row["id"]): row for row in payload["images"]}
    output = args.output
    image_output = output / args.split
    image_output.mkdir(parents=True, exist_ok=True)
    (output / "annotations").mkdir(parents=True, exist_ok=True)

    for row in image_rows.values():
        source_image = args.source / args.split / row["file_name"]
        with Image.open(source_image) as image:
            transformed = image.transpose(TRANSFORMS[args.transform])
            transformed.save(image_output / row["file_name"])

    annotations = []
    for annotation in payload["annotations"]:
        image = image_rows[int(annotation["image_id"])]
        segments = [
            transform_polygon(
                segment,
                int(image["width"]),
                int(image["height"]),
                args.transform,
            )
            for segment in annotation["segmentation"]
        ]
        annotations.append(
            {
                **annotation,
                "segmentation": segments,
                "bbox": transform_bbox(segments[0]),
            }
        )
    result = {
        key: value
        for key, value in payload.items()
        if key not in {"annotations"}
    }
    result["annotations"] = annotations
    (output / "annotations" / f"{args.split}.json").write_text(
        json.dumps(result) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(
        source_annotation,
        output / "annotations" / f"source_{args.split}.json",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "transform": args.transform,
                "split": args.split,
                "images": len(image_rows),
                "annotations": len(annotations),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

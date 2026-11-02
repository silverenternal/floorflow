#!/usr/bin/env python3
"""Build a structure-conditioned Raster2Seq proposal cache on FloorFlow samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from run_fp_raster2seq_identity import provenance
from run_fp_raster2seq_matched_edits import load_model

from floorflow.data.dataset import (
    CubiCasaRasterControlDataset,
    FloorplanSample,
    load_or_rasterize,
)
from floorflow.data.raster2seq_proposal_cache import (
    MANIFEST_VERSION,
    read_teacher_case,
    sha256_file,
    write_teacher_case,
)
from floorflow.data.raster_control import (
    footprint_center,
    load_cubicasa_floor_rgb,
    load_cubicasa_line_raster,
    non_room_control_labels,
)
from floorflow.data.svg import list_cubicasa_floors
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.raster2seq_external import (
    decode_raster2seq_proposals,
    render_raster2seq_input,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--floorflow-cache",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/external_baselines/raster2seq/cubicasa5k/checkpoint.pth"
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/raster2seq_proposal_cache_contract_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/cache/raster2seq_teacher_train256_cal64_v1",
    )
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument(
        "--input-mode",
        choices=(
            "raw_raster",
            "page_floor_crop",
            "footprint_crop_raster",
            "structural_line",
            "non_room_control",
            "cropped_non_room_control",
        ),
        default="footprint_crop_raster",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--sample-keys-file", type=Path, default=None)
    parser.add_argument("--sample-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dump-input-dir", type=Path, default=None)
    return parser.parse_args()


def _resize_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    foreground = np.any(array < 250, axis=-1)
    if foreground.any():
        ys, xs = np.nonzero(foreground)
        margin = 10
        left = max(0, int(xs.min()) - margin)
        top = max(0, int(ys.min()) - margin)
        right = min(image.width, int(xs.max()) + margin + 1)
        bottom = min(image.height, int(ys.max()) + margin + 1)
        image = image.crop((left, top, right, bottom))
    scale = min(256.0 / image.width, 256.0 / image.height)
    resized = image.resize(
        (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        ),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (256, 256), (255, 255, 255))
    canvas.paste(
        resized,
        (
            (256 - resized.width) // 2,
            (256 - resized.height) // 2,
        ),
    )
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _resize_rgb(path: Path) -> np.ndarray:
    return _resize_image(Image.open(path))


def _input_image(
    sample: FloorplanSample,
    mode: str,
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    if mode == "raw_raster":
        return _resize_rgb(sample.raster_path)
    if mode == "page_floor_crop":
        floor_ids = list_cubicasa_floors(sample.svg_path)
        _target, footprint = load_or_rasterize(
            sample,
            image_size=256,
            taxonomy=taxonomy,
            write_cache=False,
        )
        image = load_cubicasa_floor_rgb(
            sample.raster_path,
            floor_index=floor_ids.index(str(sample.floor_id)),
            floor_count=len(floor_ids),
            target_center=footprint_center(footprint),
        )
        return _resize_image(Image.fromarray(image, mode="RGB"))
    if mode == "footprint_crop_raster":
        _target, footprint = load_or_rasterize(
            sample,
            image_size=256,
            taxonomy=taxonomy,
            write_cache=False,
        )
        ys, xs = np.nonzero(footprint.astype(bool))
        if not len(xs):
            return _resize_rgb(sample.raster_path)
        image = Image.open(sample.raster_path).convert("RGB")
        margin = 10.0
        left = max(0.0, xs.min() / 255.0 * image.width - margin)
        top = max(0.0, ys.min() / 255.0 * image.height - margin)
        right = min(
            float(image.width),
            (xs.max() + 1) / 255.0 * image.width + margin,
        )
        bottom = min(
            float(image.height),
            (ys.max() + 1) / 255.0 * image.height + margin,
        )
        cropped = image.crop(
            (
                round(left),
                round(top),
                round(right),
                round(bottom),
            )
        )
        scale = min(256.0 / cropped.width, 256.0 / cropped.height)
        resized = cropped.resize(
            (
                max(1, round(cropped.width * scale)),
                max(1, round(cropped.height * scale)),
            ),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (256, 256), (255, 255, 255))
        canvas.paste(
            resized,
            (
                (256 - resized.width) // 2,
                (256 - resized.height) // 2,
            ),
        )
        return np.asarray(canvas, dtype=np.float32) / 255.0
    if mode == "structural_line":
        normalized = load_cubicasa_line_raster(
            sample.raster_path,
            image_size=256,
        )
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return np.clip(
            np.moveaxis(normalized * std + mean, 0, -1),
            0.0,
            1.0,
        )
    target, footprint = load_or_rasterize(
        sample,
        image_size=256,
        taxonomy=taxonomy,
        write_cache=False,
    )
    if mode == "cropped_non_room_control":
        from floorflow.data.raster_control import crop_floor_masks

        target, footprint = crop_floor_masks(
            target,
            footprint,
            image_size=256,
        )
    controls = non_room_control_labels(target, footprint, taxonomy)
    return (
        render_raster2seq_input(controls, taxonomy).astype(np.float32)
        / 255.0
    )


def _to_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device)


def _subset_labels(split: str, index: int) -> list[str]:
    labels = []
    if split == "val":
        if index < 64:
            labels.append("cal64")
        if 128 <= index < 192:
            labels.append("holdout64")
    return labels


def _split_manifest(root: Path, split: str) -> Path:
    return root / split / "manifest.json"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and device.index != 0:
        raise ValueError("Raster2Seq teacher generation is restricted to cuda:0")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    dataset = CubiCasaRasterControlDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.floorflow_cache,
        augment=False,
        limit=(
            None
            if args.sample_keys_file is not None or args.sample_root is not None
            else int(args.start_index) + int(args.limit)
        ),
        image_mode="native",
    )
    if args.sample_keys_file is not None or args.sample_root is not None:
        requested = (
            [
                line.strip()
                for line in args.sample_keys_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            if args.sample_keys_file is not None
            else [
                path.stem
                for path in sorted(args.sample_root.glob("*.npz"))
            ][: int(args.limit)]
        )
        lookup = {}
        for sample in dataset.samples:
            key = f"{sample.relative_case}__{sample.floor_id}"
            lookup[key] = sample
            lookup[key.replace("/", "__")] = sample
        samples = [lookup[key] for key in requested]
    else:
        samples = dataset.samples[
            int(args.start_index) : int(args.start_index) + int(args.limit)
        ]
    if len(samples) != int(args.limit):
        raise RuntimeError(
            f"expected {args.limit} {args.split} samples, found {len(samples)}"
        )
    taxonomy = FloorplanTaxonomy.read(args.floorflow_cache / "taxonomy.json")
    model, image_size = load_model(args.checkpoint, device)
    if image_size != 256:
        raise ValueError(f"expected Raster2Seq image_size=256, found {image_size}")
    cache_dir = args.output_root / args.split / "cases"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in range(0, len(samples), int(args.batch_size)):
        batch_samples = samples[start : start + int(args.batch_size)]
        batch_cache_paths = [
            cache_dir
            / (
                f"{sample.relative_case}__{sample.floor_id}"
                .replace("/", "__")
                + ".npz"
            )
            for sample in batch_samples
        ]
        needs_inference = not (
            args.skip_existing
            and all(path.is_file() for path in batch_cache_paths)
        )
        predictions = None
        if needs_inference:
            images = [
                _input_image(sample, args.input_mode, taxonomy)
                for sample in batch_samples
            ]
            if args.dump_input_dir is not None:
                args.dump_input_dir.mkdir(parents=True, exist_ok=True)
                for sample, image in zip(batch_samples, images, strict=True):
                    safe_key = (
                        f"{sample.relative_case}__{sample.floor_id}"
                        .replace("/", "__")
                    )
                    Image.fromarray(
                        np.rint(image * 255.0).clip(0, 255).astype(np.uint8)
                    ).save(args.dump_input_dir / f"{safe_key}.png")
            inputs = _to_tensor(images, device)
            with torch.inference_mode():
                predictions = model.forward_inference(inputs, use_cache=True)
        for batch_index, sample in enumerate(batch_samples):
            split_index = int(args.start_index) + start + batch_index
            sample_key = f"{sample.relative_case}__{sample.floor_id}"
            safe_key = sample_key.replace("/", "__")
            cache_path = cache_dir / f"{safe_key}.npz"
            if not (args.skip_existing and cache_path.is_file()):
                if predictions is None:
                    raise RuntimeError("teacher inference was unexpectedly skipped")
                proposals = decode_raster2seq_proposals(
                    predictions["gen_out"][batch_index],
                    predictions["pred_room_logits"][batch_index],
                    taxonomy,
                    image_size=image_size,
                )
                room_proposals = [
                    proposal
                    for proposal in proposals
                    if taxonomy.class_names[int(proposal.class_id)].startswith("room:")
                ]
                write_teacher_case(
                    cache_path,
                    sample_key=sample_key,
                    split=args.split,
                    split_index=split_index,
                    image_size=image_size,
                    polygons=[
                        proposal.polygon for proposal in room_proposals
                    ],
                    class_ids=[
                        proposal.class_id for proposal in room_proposals
                    ],
                    class_names=[
                        taxonomy.class_names[int(proposal.class_id)]
                        for proposal in room_proposals
                    ],
                    existence_scores=[
                        proposal.class_probability
                        for proposal in room_proposals
                    ],
                    sequence_length=len(predictions["gen_out"][batch_index]),
                    source_image_sha256=sha256_file(sample.raster_path),
                )
            cached = read_teacher_case(cache_path, taxonomy=taxonomy)
            rows.append(
                {
                    "sample_key": sample_key,
                    "split": args.split,
                    "split_index": split_index,
                    "subsets": _subset_labels(args.split, split_index),
                    "source_image": str(sample.raster_path),
                    "source_image_sha256": cached.source_image_sha256,
                    "cache_file": str(cache_path.relative_to(cache_dir.parent)),
                    "cache_sha256": sha256_file(cache_path),
                    "room_count": len(cached.polygons),
                    "class_names": list(cached.class_names),
                    "corner_counts": [
                        len(polygon) for polygon in cached.polygons
                    ],
                    "mean_existence": (
                        float(cached.existence_scores.mean())
                        if len(cached.existence_scores)
                        else 0.0
                    ),
                }
            )
        print(
            f"cached {min(start + len(batch_samples), len(samples))}/{len(samples)} "
            f"{args.split} samples",
            flush=True,
        )
    rows.sort(key=lambda row: row["split_index"])
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "scope": contract["scope"],
        "split": args.split,
        "start_index": int(args.start_index),
        "limit": int(args.limit),
        "cases": len(rows),
        "image_size": image_size,
        "input_protocol": {
            **contract["input_contract"],
            "mode": args.input_mode,
            "target_room_class_pixels_used": False,
            "target_non_room_channels_used": (
                args.input_mode == "non_room_control"
            ),
            "footprint_control_used": (
                args.input_mode == "footprint_crop_raster"
            ),
        },
        "teacher_input_contract": contract["input_contract"]["kind"],
        "teacher_class_outputs_are_trusted": False,
        "student_inference_target_free": True,
        "target_used_at_inference": False,
        "test_opened": False,
        "teacher_role": "predicted_control_not_ground_truth",
        "taxonomy_sha256": sha256_file(args.floorflow_cache / "taxonomy.json"),
        "contract_sha256": sha256_file(args.contract),
        "provenance": provenance(args.checkpoint),
        "rows": rows,
    }
    manifest_path = _split_manifest(args.output_root, args.split)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "split": args.split,
                "cases": len(rows),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()

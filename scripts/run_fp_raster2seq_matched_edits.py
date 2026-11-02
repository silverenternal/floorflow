#!/usr/bin/env python3
"""Run Raster2Seq independently on paired FloorFlow test503 edit controls."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.edit_detection import (
    decode_visible_controls,
    edit_detection_iou_mse,
)
from floorflow.evaluation.raster2seq_external import (
    decode_raster2seq_sequence,
    render_raster2seq_input,
    render_raster2seq_predictions,
)

RASTER2SEQ_ROOT = Path("third_party/Raster2Seq").resolve()
sys.path.insert(0, str(RASTER2SEQ_ROOT / "models" / "ops"))
sys.path.insert(0, str(RASTER2SEQ_ROOT / "diff_ras"))
sys.path.insert(0, str(RASTER2SEQ_ROOT))

from datasets.discrete_tokenizer import DiscreteTokenizer
from models import build_model

EDITS = {
    "room_type": "fp_local_edit_room_type_test503_v1",
    "room_add": "fp_local_edit_room_add_test503_v1",
    "room_remove": "fp_local_edit_room_remove_test503_v1",
    "opening_delete": "fp_local_edit_opening_delete_test503_v1",
    "furniture_delete": "fp_local_edit_furniture_delete_test503_v1",
}


@dataclass(frozen=True)
class EditPair:
    operation: str
    name: str
    before_image: np.ndarray
    after_image: np.ndarray
    target_before: np.ndarray
    target_after: np.ndarray
    valid: np.ndarray
    outside: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/external_baselines/raster2seq/cubicasa5k/checkpoint.pth"
        ),
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("outputs/control_contracts"),
    )
    parser.add_argument(
        "--raster-root",
        type=Path,
        default=Path(
            "outputs/eval/fp_hard_metrics_full_test503_seed19_v1/rasters"
        ),
    )
    parser.add_argument(
        "--edit-cache",
        type=Path,
        default=Path("outputs/cache/fp_edit_detection_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/external_baselines/raster2seq/test503_matched_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_raster(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def region_mask(shape: tuple[int, int], normalized: list[float]) -> np.ndarray:
    height, width = shape
    x, y, box_width, box_height = normalized
    left = max(0, min(width, round(float(x) * width)))
    top = max(0, min(height, round(float(y) * height)))
    right = max(left + 1, min(width, round((x + box_width) * width)))
    bottom = max(top + 1, min(height, round((y + box_height) * height)))
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def build_pairs(
    args: argparse.Namespace,
    taxonomy: FloorplanTaxonomy,
) -> list[EditPair]:
    source_manifest_path = (
        args.control_root / "oracle_v1" / "test" / "manifest.json"
    )
    source_manifest = read_json(source_manifest_path)
    before_controls = {
        row["name"]: Path(row["file"]) for row in source_manifest["rows"]
    }
    before_label_cache: dict[str, np.ndarray] = {}
    before_image_cache: dict[str, np.ndarray] = {}
    pairs: list[EditPair] = []
    for operation, directory in EDITS.items():
        manifest = read_json(
            args.control_root / directory / "test" / "manifest.json"
        )
        for record in manifest["rows"]:
            if not record["edited"]:
                continue
            name = record["name"]
            safe_name = name.replace("/", "__")
            raster = load_raster(args.raster_root / f"{safe_name}.npz")
            footprint = raster["footprint"].astype(bool)
            target_before = raster["target"].astype(np.uint8, copy=True)
            with np.load(args.edit_cache / operation / f"{safe_name}.npz") as payload:
                changed = payload["changed"].astype(bool)
                after_control = payload["after_visible"].astype(np.uint8)
            target_after = target_before.copy()
            target_after[changed] = after_control[changed]

            if name not in before_label_cache:
                before_label_cache[name] = decode_visible_controls(
                    before_controls[name],
                    footprint,
                    taxonomy,
                )
            before_labels = before_label_cache[name]
            after_labels = after_control
            if name not in before_image_cache:
                before_image_cache[name] = render_raster2seq_input(
                    before_labels,
                    taxonomy,
                )
            edit_region = region_mask(
                target_before.shape,
                record["edit_region_normalized"],
            ) & footprint
            wall = (
                (target_before == taxonomy.class_id("wall"))
                | (target_after == taxonomy.class_id("wall"))
            )
            wall = binary_dilation(wall, iterations=1)
            valid = footprint & ~wall
            pairs.append(
                EditPair(
                    operation=operation,
                    name=name,
                    before_image=before_image_cache[name],
                    after_image=render_raster2seq_input(after_labels, taxonomy),
                    target_before=target_before,
                    target_after=target_after,
                    valid=valid,
                    outside=valid & ~edit_region,
                )
            )
            if args.limit is not None and len(pairs) >= args.limit:
                return pairs
    return pairs


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_args = copy.deepcopy(checkpoint["args"])
    model_args.vocab_size = None
    tokenizer = DiscreteTokenizer(
        model_args.num_bins,
        model_args.seq_len,
        add_cls=model_args.add_cls_token,
    )
    model_args.vocab_size = len(tokenizer)
    model = build_model(model_args, train=False, tokenizer=tokenizer)
    missing, _unexpected = model.load_state_dict(
        checkpoint["ema"],
        strict=False,
    )
    if missing:
        raise RuntimeError(f"checkpoint is missing tensors: {missing[:5]}")
    return model.to(device).eval(), model_args.image_size


def to_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.stack(images))
        .permute(0, 3, 1, 2)
        .float()
        .div(255.0)
        .to(device)
    )


def room_support_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> float:
    predicted = np.isin(prediction, taxonomy.room_ids)
    expected = np.isin(target, taxonomy.room_ids)
    union = np.logical_or(predicted, expected).sum()
    return (
        1.0
        if union == 0
        else float(np.logical_and(predicted, expected).sum() / union)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    pairs = build_pairs(args, taxonomy)
    if not pairs:
        raise RuntimeError("no edit pairs were constructed")
    device = torch.device(args.device)
    model, image_size = load_model(args.checkpoint, device)
    rows: list[dict] = []
    prediction_root = args.output_dir / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        images = [
            image
            for pair in batch
            for image in (pair.before_image, pair.after_image)
        ]
        inputs = to_tensor(images, device)
        with torch.no_grad():
            outputs = model.forward_inference(inputs, use_cache=True)
        for index, pair in enumerate(batch):
            path = prediction_root / pair.operation / f"{pair.name.replace('/', '__')}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not (args.skip_existing and path.is_file()):
                predictions = []
                for offset in (2 * index, 2 * index + 1):
                    polygons, labels = decode_raster2seq_sequence(
                        outputs["gen_out"][offset],
                        outputs["pred_room_logits"][offset],
                        taxonomy,
                        image_size=image_size,
                    )
                    predictions.append(
                        render_raster2seq_predictions(
                            polygons,
                            labels,
                            pair.target_before.shape,
                            taxonomy,
                        )
                    )
                np.savez_compressed(
                    path,
                    prediction_before=predictions[0],
                    prediction_after=predictions[1],
                )
            with np.load(path) as payload:
                before = payload["prediction_before"]
                after = payload["prediction_after"]
            delta_iou, delta_mse = edit_detection_iou_mse(
                pair.target_before,
                pair.target_after,
                before,
                after,
                pair.valid,
            )
            outside_equal = before[pair.outside] == after[pair.outside]
            rows.append(
                {
                    "operation": pair.operation,
                    "name": pair.name,
                    "delta_iou": delta_iou,
                    "delta_mse": delta_mse,
                    "outside_exact_ratio": (
                        float(outside_equal.mean())
                        if outside_equal.size
                        else 1.0
                    ),
                    "outside_changed_pixels": int((~outside_equal).sum()),
                    "before_room_support_iou": room_support_iou(
                        before,
                        pair.target_before,
                        taxonomy,
                    ),
                    "after_room_support_iou": room_support_iou(
                        after,
                        pair.target_after,
                        taxonomy,
                    ),
                    "prediction_file": str(path),
                }
            )
        print(
            f"processed {min(start + len(batch), len(pairs))}/{len(pairs)} pairs",
            flush=True,
        )

    summary = {}
    for operation in EDITS:
        operation_rows = [
            row for row in rows if row["operation"] == operation
        ]
        if not operation_rows:
            continue
        summary[operation] = {
            "cases": len(operation_rows),
            **{
                key: float(np.mean([row[key] for row in operation_rows]))
                for key in (
                    "delta_iou",
                    "delta_mse",
                    "outside_exact_ratio",
                    "outside_changed_pixels",
                    "before_room_support_iou",
                    "after_room_support_iou",
                )
            },
        }
    report = {
        "status": "complete",
        "method": "Raster2Seq",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "repository": str(RASTER2SEQ_ROOT),
        "repository_commit": "a6c4e27a68d11d7a459f6e4a2601fd887227dd1a",
        "condition_mode": "matched_coarse_control_line_render",
        "input_contract": (
            "The same FloorFlow before/after coarse controls are rendered as "
            "neutral architectural line drawings; Raster2Seq independently "
            "regenerates each side without access to edit metadata."
        ),
        "evaluator": (
            "HouseMind B.3 Delta-IoU/MSE with FloorFlow wall exclusion and "
            "the same test503 edit rows."
        ),
        "seed": 42,
        "batch_size": args.batch_size,
        "image_size": image_size,
        "pairs": len(pairs),
        "summary": summary,
        "rows": rows,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

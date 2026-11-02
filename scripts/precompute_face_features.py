#!/usr/bin/env python3
"""Precompute per-wall-face evidence features from a frozen floorplan model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.heads.face_semantics import pool_face_evidence
from floorflow.data.geometry_cache import load_geometry_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_geometry_v1",
    )
    parser.add_argument(
        "--face-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_faces_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_face_features_v1",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/reports/face_features_v1.json",
    )
    return parser.parse_args()


def _descriptors(masks: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    rows = []
    count, height, width = masks.shape
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    for index, mask in enumerate(masks):
        area = max(1.0, float(mask.sum()))
        rows.append(
            [
                area / float(height * width),
                float((mask * xs).sum() / area / max(1, width - 1)),
                float((mask * ys).sum() / area / max(1, height - 1)),
                float(mask.any(axis=1).sum() / height),
                float(mask.any(axis=0).sum() / width),
                float(adjacency[index].sum()),
            ]
        )
    return np.asarray(rows, dtype=np.float32).reshape(count, 6)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        device,
        weight_source="model",
    )
    model.eval()
    rows = []
    for split in args.splits:
        sources = sorted(
            (args.geometry_root / split).rglob("*.geometry.npz")
        )
        for completed, source in enumerate(sources, start=1):
            relative = source.relative_to(args.geometry_root)
            face_path = args.face_root / relative.with_name(
                relative.name.replace(".geometry.npz", ".faces.npz")
            )
            target_path = args.output_root / relative.with_name(
                relative.name.replace(".geometry.npz", ".face_features.npz")
            )
            if target_path.is_file() and not args.overwrite:
                rows.append(
                    {
                        "case": str(relative),
                        "split": split,
                        "status": "reused",
                    }
                )
                continue
            try:
                geometry = load_geometry_cache(source)
                with np.load(face_path) as face_payload:
                    masks = face_payload["masks"]
                    adjacency = face_payload["adjacency"]
                    classes = face_payload["classes"]
                    purity = face_payload["purity"]
                if not len(masks):
                    rows.append(
                        {
                            "case": str(relative),
                            "split": split,
                            "status": "empty",
                            "faces": 0,
                        }
                    )
                    continue
                target = torch.from_numpy(geometry["target"]).to(device)
                footprint = torch.from_numpy(geometry["footprint"]).to(device)
                floor_id = int(source.name.split("_")[1].split(".")[0])
                zero = torch.zeros(
                    1,
                    model.target_channels,
                    *target.shape,
                    device=device,
                    dtype=torch.float32,
                )
                with torch.no_grad(), torch.autocast(
                    device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    features = model._encode_features(
                        zero,
                        torch.zeros(1, device=device),
                        footprint[None],
                        floor_ids=torch.tensor([floor_id], device=device),
                    )
                    semantic_logits = model.semantic_endpoint(features)
                    room_prior = semantic_logits[
                        :,
                        model.room_class_ids.to(device) - 1,
                    ]
                    room_logits = model.room_partition(features, room_prior)
                    objects = model.structured_objects(features.detach())
                    opening_scores = torch.cat(
                        [objects["opening_background_logits"][:, None], objects["opening_logits"]],
                        dim=1,
                    )
                    furniture_scores = torch.cat(
                        [objects["furniture_background_logits"][:, None], objects["furniture_logits"]],
                        dim=1,
                    )
                    evidence = torch.cat(
                        [
                            features,
                            room_logits.softmax(dim=1),
                            opening_scores.softmax(dim=1)[:, 1:],
                            furniture_scores.softmax(dim=1)[:, 1:],
                            objects["wall_logits"].sigmoid()[:, None],
                        ],
                        dim=1,
                    ).float()
                pooled = pool_face_evidence(evidence[0], list(masks))
                node_features = np.concatenate(
                    [
                        pooled.detach().cpu().numpy(),
                        _descriptors(masks, adjacency),
                    ],
                    axis=1,
                ).astype(np.float32)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = target_path.with_suffix(f".tmp-{os.getpid()}.npz")
                np.savez_compressed(
                    temporary,
                    nodes=node_features,
                    adjacency=adjacency,
                    classes=classes,
                    purity=purity,
                )
                os.replace(temporary, target_path)
                rows.append(
                    {
                        "case": str(relative),
                        "split": split,
                        "status": "written",
                        "faces": int(len(masks)),
                        "features": int(node_features.shape[1]),
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "case": str(relative),
                        "split": split,
                        "status": "error",
                        "error": repr(error),
                    }
                )
            if completed % 200 == 0 or completed == len(sources):
                print(f"[{split}] {completed}/{len(sources)}", flush=True)
    report = {
        "version": "face-features-v1",
        "checkpoint": str(args.checkpoint),
        "splits": {
            split: sum(row["split"] == split for row in rows)
            for split in args.splits
        },
        "errors": sum(row["status"] == "error" for row in rows),
        "faces": sum(int(row.get("faces") or 0) for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("splits", "errors", "faces")},
            indent=2,
        )
    )
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

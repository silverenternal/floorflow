#!/usr/bin/env python3
"""Train image-aware semantic assignment on frozen native room support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_raster_predicted_control_bridge import RasterPredictedControlDataset

from floorflow.data.raster_control import load_cubicasa_structure_mask
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.component_image_semantics import native_component_records
from floorflow.heads.component_semantic import (
    ComponentSemanticClassifier,
    assign_component_classes,
)
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.representation.core import extract_room_instances

ROOT = Path(__file__).resolve().parents[1]


class StructureAwareRasterDataset(RasterPredictedControlDataset):
    def structure_mask(self, name: str) -> torch.Tensor:
        category, case_id, floor = name.split("__", maxsplit=2)
        floor_number = floor.rsplit("-", 1)[-1]
        path = (
            self.data_root
            / category
            / case_id
            / f"F{floor_number}_scaled.png"
        )
        return torch.from_numpy(
            load_cubicasa_structure_mask(
                path,
                image_size=self.image_size,
            )
        ).bool()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/native_component_semantic_screen256_v1",
    )
    return parser.parse_args()


@torch.inference_mode()
def extract_records(model, loader, taxonomy, room_size, device):
    features = []
    targets = []
    for target, footprint, _, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        logits = model(footprint, control_image=image)
        footprint_room = footprint.long()
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        labels = room_ids[logits.argmax(dim=1)] * footprint_room
        probability = logits.softmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        target_np = target.cpu().numpy()
        for index in range(target.shape[0]):
            structure = loader.dataset.structure_mask(names[index]).to(device)
            case_features, case_targets, _ids = native_component_records(
                labels_np[index],
                target_np[index],
                probability[index],
                image[index].cpu().numpy(),
                footprint_room[index].cpu().numpy(),
                taxonomy.room_ids,
                structure_mask=structure.cpu().numpy(),
            )
            features.append(case_features)
            targets.append(case_targets)
    return np.concatenate(features), np.concatenate(targets)


@torch.inference_mode()
def evaluate(model, classifier, loader, taxonomy, room_size, device):
    arms = {
        "baseline": [],
        "component_classifier": [],
    }
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    for target, footprint, _, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        logits = model(footprint, control_image=image)
        labels = room_ids[logits.argmax(dim=1)]
        probability = logits.softmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        target_np = target.cpu().numpy()
        footprint_np = footprint.cpu().numpy()
        for index in range(target.shape[0]):
            structure = loader.dataset.structure_mask(names[index]).to(device)
            features, _targets, component_ids = native_component_records(
                labels_np[index],
                target_np[index],
                probability[index],
                image[index].cpu().numpy(),
                footprint_np[index],
                taxonomy.room_ids,
                structure_mask=structure.cpu().numpy(),
            )
            class_probability = (
                classifier(torch.from_numpy(features).to(device))
                .softmax(dim=-1)
                .cpu()
                .numpy()
                if len(features)
                else np.zeros((0, len(taxonomy.room_ids)), dtype=np.float32)
            )
            assigned = assign_component_classes(
                labels_np[index] * footprint_np[index],
                class_probability,
                component_ids,
                taxonomy.room_ids,
                structure_mask=structure.cpu().numpy(),
            ) * footprint_np[index]
            for arm, prediction in (
                ("baseline", labels_np[index] * footprint_np[index]),
                ("component_classifier", assigned),
            ):
                target_instances = extract_room_instances(
                    target_np[index],
                    taxonomy,
                    min_area=8,
                )
                suite = room_metric_suite(
                    prediction,
                    target_np[index],
                    footprint_np[index],
                    taxonomy,
                    predicted_room_support=np.isin(
                        prediction,
                        taxonomy.room_ids,
                    ),
                    predicted_instance_support=np.isin(
                        prediction,
                        taxonomy.room_ids,
                    ),
                    target_room_support=np.isin(
                        target_np[index],
                        taxonomy.room_ids,
                    ),
                )
                instance = suite["matched_instance"]
                arms[arm].append(
                    {
                        "room_macro_iou": suite["visible_semantic"][
                            "room_macro_iou"
                        ],
                        "instance_f1": instance[
                            "instance_f1_at_threshold"
                        ],
                        "type_accuracy": instance[
                            "matched_type_accuracy"
                        ],
                        "class_constrained_iou": instance[
                            "matched_class_constrained_iou"
                        ],
                        "count_abs_error": abs(
                            int(instance["predicted_instances"])
                            - len(target_instances)
                        ),
                    }
                )
    return {
        arm: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        for arm, rows in arms.items()
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    predicted_root = ROOT / "outputs/control_contracts/model_predicted_full_v1"
    train = StructureAwareRasterDataset(
        predicted_root,
        "train",
        args.cache_root,
        args.data_root,
        limit=args.limit_train,
    )
    validation = StructureAwareRasterDataset(
        predicted_root,
        "val",
        args.cache_root,
        args.data_root,
        limit=args.limit_val,
    )
    room_model = load_raster_room_model(args.room_checkpoint, taxonomy, device)
    room_model.eval()
    common = {
        "batch_size": 8,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train, shuffle=False, **common)
    val_loader = DataLoader(validation, shuffle=False, **common)
    train_features, train_targets = extract_records(
        room_model,
        train_loader,
        taxonomy,
        args.room_size,
        device,
    )
    val_features, val_targets = extract_records(
        room_model,
        val_loader,
        taxonomy,
        args.room_size,
        device,
    )
    counts = np.bincount(train_targets, minlength=len(taxonomy.room_ids))
    class_weights = torch.from_numpy(
        (1.0 / np.sqrt(np.maximum(counts, 1))).astype(np.float32)
    ).to(device)
    class_weights /= class_weights.mean().clamp_min(1.0e-6)
    classifier = ComponentSemanticClassifier(
        train_features.shape[1],
        len(taxonomy.room_ids),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    history = []
    best = None
    best_score = -1.0
    for epoch in range(args.epochs):
        classifier.train()
        permutation = np.random.default_rng(epoch).permutation(len(train_features))
        total = 0.0
        for start in range(0, len(train_features), args.batch_size):
            index = permutation[start : start + args.batch_size]
            logits = classifier(
                torch.from_numpy(train_features[index]).to(device)
            )
            loss = F.cross_entropy(
                logits,
                torch.from_numpy(train_targets[index]).to(device),
                weight=class_weights,
                label_smoothing=0.05,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        validation_logits = classifier(
            torch.from_numpy(val_features).to(device)
        )
        component_accuracy = float(
            (validation_logits.argmax(dim=-1).cpu().numpy() == val_targets).mean()
        )
        arms = evaluate(
            room_model,
            classifier,
            val_loader,
            taxonomy,
            args.room_size,
            device,
        )
        score = (
            arms["component_classifier"]["room_macro_iou"]
            + 0.25 * arms["component_classifier"]["type_accuracy"]
        )
        row = {
            "epoch": epoch,
            "train_loss": total / max(1, len(train_features)),
            "component_accuracy": component_accuracy,
            "arms": arms,
            "selection_score": score,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            best = {
                "model": classifier.state_dict(),
                "feature_size": int(train_features.shape[1]),
                "num_classes": len(taxonomy.room_ids),
                "class_ids": list(taxonomy.room_ids),
                "config": vars(args),
                "metrics": row,
            }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best, args.output_dir / "best.pt")
    report = {
        "status": "complete",
        "scope": "native raster component semantic screen",
        "target_used_at_inference": False,
        "train_cases": len(train),
        "validation_cases": len(validation),
        "best_selection_score": best_score,
        "history": history,
        "config": vars(args),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

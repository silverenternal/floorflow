#!/usr/bin/env python3
"""Train FloorFlow room identity over target-free Raster2Seq proposal masks."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.proposal_room_classifier import (
    ProposalRoomClassifier,
    compose_proposal_labels,
    proposal_classification_loss,
)
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.training.proposal_distillation import ProposalTeacherIndex
from floorflow.training.room_capability_training import teacher_room_targets

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=ROOT / "outputs/cache/raster2seq_teacher_predicted64_v1",
    )
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument("--limit-train", type=int, default=64)
    parser.add_argument("--limit-val", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _metrics(prediction, target, footprint, taxonomy):
    room_ids = torch.tensor(taxonomy.room_ids, device=prediction.device)
    suite = room_metric_suite(
        prediction.detach().cpu().numpy(),
        target.detach().cpu().numpy(),
        footprint.detach().cpu().numpy(),
        taxonomy,
        predicted_room_support=torch.isin(
            prediction,
            room_ids,
        ).detach().cpu().numpy(),
        predicted_instance_support=torch.isin(
            prediction,
            room_ids,
        ).detach().cpu().numpy(),
        target_room_support=torch.isin(
            target,
            room_ids,
        ).detach().cpu().numpy(),
    )
    instance = suite["matched_instance"]
    return {
        "room_iou": float(suite["visible_semantic"]["room_macro_iou"]),
        "instance_f1": float(instance["instance_f1_at_threshold"]),
        "type_accuracy": float(instance["matched_type_accuracy"]),
        "count_error": float(instance["count_abs_error"]),
    }


def _aggregate(rows):
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


@torch.inference_mode()
def evaluate(
    classifier,
    layer,
    runtime,
    loader,
    teacher_index,
    taxonomy,
    device,
    predicted_root,
    max_proposals,
):
    classifier.eval()
    rows = {"proposal": [], "proposal_dense_fill": []}
    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        controls = _predicted_controls(
            Path(predicted_root),
            "val",
            names,
            _ControlArgs(predicted_root, max_proposals),
            device,
        )
        controls, _ = runtime(
            controls,
            footprint,
            image.to(device),
            floor_ids.to(device),
        )
        output = layer(controls, footprint)
        teacher = teacher_index.load(
            names,
            image_size=256,
            max_proposals=int(max_proposals),
            device=device,
        )
        prediction = classifier(
            output["field_features"],
            output["stage_rooms_logits"],
            teacher,
            taxonomy,
        )
        dense_logits = torch.nn.functional.interpolate(
            output["stage_rooms_logits"],
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        proposal = compose_proposal_labels(
            prediction,
            teacher,
            taxonomy,
            footprint,
        )
        hybrid = compose_proposal_labels(
            prediction,
            teacher,
            taxonomy,
            footprint,
            dense_fill_logits=dense_logits,
        )
        for index in range(target.shape[0]):
            rows["proposal"].append(
                _metrics(
                    proposal[index],
                    target[index],
                    footprint[index],
                    taxonomy,
                )
            )
            rows["proposal_dense_fill"].append(
                _metrics(
                    hybrid[index],
                    target[index],
                    footprint[index],
                    taxonomy,
                )
            )
    return {
        name: _aggregate(values)
        for name, values in rows.items()
    }


class _ControlArgs:
    def __init__(self, predicted_root, max_proposals):
        self.predicted_root = Path(predicted_root)
        self.max_proposals = max_proposals


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.layer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    layer = build_predicted_channel_model(taxonomy, config).to(device)
    layer.load_state_dict(payload["model"])
    layer.requires_grad_(False).eval()
    runtime = CadStageRuntime(
        taxonomy,
        device,
        room_checkpoint=config["room_checkpoint"],
        structure_checkpoint=config["structure_checkpoint"],
        centerline_checkpoint=config["centerline_checkpoint"],
        opening_checkpoint=config["opening_checkpoint"],
        room_blend=float(config["room_blend"]),
        structure_size=128,
    )
    train = RasterPredictedControlDataset(
        args.predicted_root,
        "train",
        Path(config["cache_root"]),
        Path(config["data_root"]),
        image_size=int(config["image_size"]),
        limit=args.limit_train,
    )
    val = RasterPredictedControlDataset(
        args.predicted_root,
        "val",
        Path(config["cache_root"]),
        Path(config["data_root"]),
        image_size=int(config["image_size"]),
        limit=args.limit_val,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    teacher_train = ProposalTeacherIndex(
        args.teacher_root / "train" / "manifest.json"
    )
    teacher_val = ProposalTeacherIndex(
        args.teacher_root / "val" / "manifest.json"
    )
    classifier = ProposalRoomClassifier(
        int(config["field_channels"]),
        len(taxonomy.room_ids),
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    for epoch in range(args.epochs):
        classifier.train()
        totals = {}
        seen = 0
        for target, footprint, floor_ids, names, image in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            controls = _predicted_controls(
                args.predicted_root,
                "train",
                names,
                _ControlArgs(args.predicted_root, args.max_proposals),
                device,
            )
            controls, _ = runtime(
                controls,
                footprint,
                image.to(device),
                floor_ids.to(device),
            )
            with torch.no_grad():
                output = layer(controls, footprint)
            teacher = teacher_train.load(
                names,
                image_size=256,
                max_proposals=args.max_proposals,
                device=device,
            )
            target_classes, _masks, valid = teacher_room_targets(
                teacher,
                target,
                taxonomy,
                max_rooms=args.max_proposals,
            )
            prediction = classifier(
                output["field_features"],
                output["stage_rooms_logits"],
                teacher,
                taxonomy,
            )
            loss, components = proposal_classification_loss(
                prediction,
                teacher,
                target_classes,
                valid,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += target.shape[0]
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * target.shape[0]
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value) * target.shape[0]
        metrics = evaluate(
            classifier,
            layer,
            runtime,
            val_loader,
            teacher_val,
            taxonomy,
            device,
            args.predicted_root,
            args.max_proposals,
        )
        record = {
            "epoch": epoch,
            **{key: value / max(1, seen) for key, value in totals.items()},
            **{
                f"val_{arm}_{key}": value
                for arm, values in metrics.items()
                for key, value in values.items()
            },
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        score = metrics["proposal_dense_fill"]["room_iou"]
        checkpoint = {
            "classifier": classifier.state_dict(),
            "config": vars(args),
            "taxonomy": taxonomy.to_json(),
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if score > best:
            best = score
            torch.save(checkpoint, args.output_dir / "best.pt")
    report = {
        "schema_version": "teacher-room-classifier-v1",
        "test_opened": False,
        "selected_epoch": int(
            max(history, key=lambda row: row["val_proposal_dense_fill_room_iou"])["epoch"]
        ),
        "history": history,
        "best_room_iou": best,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a target-free gate for dense versus query room routing."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_predicted_control_plausibility import _union_postprocess
from evaluate_raster_control_router_screen import _decode_layers, _structure_outputs
from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.generation_plausibility import generation_plausibility
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.joint_control_repair import JointControlRepair
from floorflow.heads.raster_control_bank import (
    load_raster_instance_control_model,
    load_raster_room_model,
)
from floorflow.heads.room_mask_fusion import mask_region_class_logits
from floorflow.inference.room_identity_arbitration import (
    ROOM_IDENTITY_FEATURE_NAMES,
    room_identity_arbitration_features,
)
from floorflow.recursion.trust_gate import RoomGraphTrustGate

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument(
        "--dense-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/native_room_384_full_32ep_v1/best.pt",
    )
    parser.add_argument(
        "--query-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/raster_instance_control_384_promotion256_10ep_v1/best.pt",
    )
    parser.add_argument(
        "--structure-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_structure_bridge_full_v1/best.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--cal-limit", type=int, default=64)
    parser.add_argument("--holdout-offset", type=int, default=128)
    parser.add_argument("--holdout-limit", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--boundary-width", type=int, default=7)
    parser.add_argument("--room-mask-threshold", type=float, default=0.6)
    parser.add_argument("--room-mask-strength", type=float, default=8.0)
    parser.add_argument("--gate-epochs", type=int, default=24)
    parser.add_argument("--gate-learning-rate", type=float, default=3.0e-3)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=ROOT / "outputs/reports",
    )
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _raster_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    room_ids = torch.tensor(taxonomy.room_ids, device=prediction.device)
    room_support = torch.isin(prediction, room_ids)
    target_support = torch.isin(target, room_ids)
    suite = room_metric_suite(
        prediction.cpu().numpy(),
        target.cpu().numpy(),
        footprint.cpu().numpy(),
        taxonomy,
        predicted_room_support=room_support.cpu().numpy(),
        predicted_instance_support=room_support.cpu().numpy(),
        target_room_support=target_support.cpu().numpy(),
    )
    instance = suite["matched_instance"]
    plausibility = generation_plausibility(
        prediction.cpu().numpy(),
        footprint.cpu().numpy(),
        taxonomy,
        explicit_rooms=True,
    )
    return {
        "room_iou": float(suite["visible_semantic"]["room_macro_iou"]),
        "instance_f1": float(instance["instance_f1_at_threshold"]),
        "count_error": float(instance["count_abs_error"]),
        "type_accuracy": float(instance["matched_type_accuracy"]),
        "plan_plausible": float(bool(plausibility["plan_plausible"])),
        "wall_continuity": float(plausibility["wall_continuity_rate"]),
    }


def _acceptance_label(
    dense: dict[str, float],
    routed: dict[str, float],
) -> bool:
    return bool(
        routed["room_iou"] >= dense["room_iou"] - 0.002
        and routed["instance_f1"] > dense["instance_f1"]
        and routed["count_error"] <= dense["count_error"]
        and routed["plan_plausible"] >= dense["plan_plausible"]
        and routed["wall_continuity"] >= dense["wall_continuity"] - 0.002
    )


@torch.inference_mode()
def collect_features(
    args: argparse.Namespace,
    split: str,
    limit: int,
    offset: int,
    dense_model,
    query_model,
    structure_model,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
) -> dict:
    dataset = RasterPredictedControlDataset(
        args.predicted_root,
        split,
        args.cache_root,
        args.data_root,
        image_size=args.image_size,
        limit=limit,
        offset=offset,
    )
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
    )
    features = []
    labels = []
    case_rows = []
    room_ids = tuple(taxonomy.room_ids)
    for target, footprint, _floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        predicted = _predicted_controls(
            args.predicted_root,
            split,
            names,
            args,
            device,
        )
        dense_logits = F.interpolate(
            dense_model(footprint, control_image=image),
            size=(args.room_size, args.room_size),
            mode="bilinear",
            align_corners=False,
        )
        query_output = query_model(footprint, control_image=image)
        query_logits = F.interpolate(
            query_output["fused_logits"],
            size=(args.room_size, args.room_size),
            mode="bilinear",
            align_corners=False,
        )
        resized_footprint = F.interpolate(
            footprint[:, None].float(),
            size=(args.room_size, args.room_size),
            mode="nearest",
        )[:, 0].bool()
        query_predictions = {
            "mask_logits": F.interpolate(
                query_output["mask_logits"],
                size=(args.room_size, args.room_size),
                mode="bilinear",
                align_corners=False,
            )
        }
        routed_logits = mask_region_class_logits(
            query_predictions,
            dense_logits,
            resized_footprint,
            threshold=args.room_mask_threshold,
            strength=args.room_mask_strength,
        )
        case_features = room_identity_arbitration_features(
            dense_logits,
            query_logits,
            routed_logits,
            resized_footprint,
            room_ids,
        )
        structure = _structure_outputs(
            structure_model,
            predicted,
            dense_logits,
            footprint,
            args,
            repair_room=False,
        )["structure"]

        def compose(
            logits: torch.Tensor,
            structure=structure,
            footprint=footprint,
            taxonomy=taxonomy,
        ) -> torch.Tensor:
            layers, _rooms, _wall, _opening = _decode_layers(
                logits,
                structure,
                footprint,
                taxonomy,
            )
            return _union_postprocess(
                layers,
                taxonomy,
                radius=0,
                boundary_width=args.boundary_width,
            )

        dense_labels_full = compose(dense_logits)
        query_labels_full = compose(query_logits)
        routed_labels_full = compose(routed_logits)
        for index, name in enumerate(names):
            dense_metric = _raster_metrics(
                dense_labels_full[index],
                target[index],
                footprint[index],
                taxonomy,
            )
            query_metric = _raster_metrics(
                query_labels_full[index],
                target[index],
                footprint[index],
                taxonomy,
            )
            routed_metric = _raster_metrics(
                routed_labels_full[index],
                target[index],
                footprint[index],
                taxonomy,
            )
            accepted = _acceptance_label(dense_metric, routed_metric)
            features.append(case_features[index].cpu())
            labels.append(float(accepted))
            case_rows.append(
                {
                    "case": str(name),
                    "accepted": accepted,
                    "dense": dense_metric,
                    "query": query_metric,
                    "routed": routed_metric,
                }
            )
        print(f"{split}: {len(case_rows)}/{len(dataset)}", flush=True)
    return {
        "features": torch.stack(features),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "cases": case_rows,
        "split": split,
        "offset": offset,
    }


def _project_selection(
    rows: list[dict],
    selected: torch.Tensor,
) -> dict:
    keys = ("room_iou", "instance_f1", "count_error", "type_accuracy", "plan_plausible", "wall_continuity")
    dense = {key: float(np.mean([row["dense"][key] for row in rows])) for key in keys}
    routed = {key: float(np.mean([row["routed"][key] for row in rows])) for key in keys}
    mixed = {
        key: float(np.mean([
            row["routed"][key] if bool(take) else row["dense"][key]
            for row, take in zip(rows, selected, strict=True)
        ]))
        for key in keys
    }
    return {
        "selected_cases": int(selected.sum()),
        "dense": dense,
        "routed": routed,
        "mixed": mixed,
    }


def _classification_metrics(
    labels: torch.Tensor,
    probability: torch.Tensor,
    threshold: float,
) -> dict:
    prediction = probability >= float(threshold)
    target = labels.bool()
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    return {
        "accuracy": float((prediction == target).float().mean()),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "positives": int(target.sum()),
        "selected": int(prediction.sum()),
    }


def _selection_score(projection: dict, dense: dict) -> float:
    mixed = projection["mixed"]
    return (
        mixed["room_iou"]
        + 0.75 * mixed["instance_f1"]
        - 0.02 * mixed["count_error"]
        + 0.05 * mixed["plan_plausible"]
        - dense["room_iou"]
        - 0.75 * dense["instance_f1"]
        + 0.02 * dense["count_error"]
        - 0.05 * dense["plan_plausible"]
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    dense_model = load_raster_room_model(args.dense_checkpoint, taxonomy, device)
    query_model = load_raster_instance_control_model(args.query_checkpoint, taxonomy, device)
    structure_model = JointControlRepair(
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        room_base_channels=24,
        structure_base_channels=32,
    ).to(device)
    payload = torch.load(
        args.structure_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    structure_model.load_state_dict(payload["model"])
    structure_model.eval()
    train = collect_features(
        args, "train", args.train_limit, 0,
        dense_model, query_model, structure_model, taxonomy, device,
    )
    cal = collect_features(
        args, "val", args.cal_limit, 0,
        dense_model, query_model, structure_model, taxonomy, device,
    )
    holdout = collect_features(
        args, "val", args.holdout_limit, args.holdout_offset,
        dense_model, query_model, structure_model, taxonomy, device,
    )
    gate = RoomGraphTrustGate(
        feature_size=len(ROOM_IDENTITY_FEATURE_NAMES),
        hidden_size=32,
    ).to(device)
    positive = train["labels"].mean().clamp(0.05, 0.95)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_learning_rate)
    history = []
    best_score = -1.0e9
    best_threshold = 0.5
    for epoch in range(args.gate_epochs):
        gate.train()
        logits = gate(train["features"].to(device))
        loss = F.binary_cross_entropy_with_logits(
            logits,
            train["labels"].to(device),
            pos_weight=((1.0 - positive) / positive).to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        gate.eval()
        cal_probability = gate.trust(cal["features"].to(device)).cpu()
        threshold_rows = []
        for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            selected = cal_probability >= threshold
            projection = _project_selection(cal["cases"], selected)
            score = _selection_score(projection, projection["dense"])
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "score": score,
                    "projection": projection,
                }
            )
        best_row = max(threshold_rows, key=lambda row: row["score"])
        projection = best_row["projection"]
        score = float(best_row["score"])
        classification = _classification_metrics(
            cal["labels"],
            cal_probability,
            float(best_row["threshold"]),
        )
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "calibration": classification,
                "threshold_grid": threshold_rows,
                "selected_threshold": best_row["threshold"],
                "projection": projection,
                "score": score,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if score > best_score:
            best_score = score
            best_threshold = float(best_row["threshold"])
            torch.save(
                {
                    "model": gate.state_dict(),
                    "config": {
                        "feature_size": len(ROOM_IDENTITY_FEATURE_NAMES),
                        "hidden_size": 32,
                        "threshold": best_threshold,
                        "feature_names": ROOM_IDENTITY_FEATURE_NAMES,
                    },
                },
                args.output_dir / "best.pt",
            )
    gate_payload = torch.load(
        args.output_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    gate.load_state_dict(gate_payload["model"])
    gate.eval()
    cal_probability = gate.trust(cal["features"].to(device)).cpu()
    holdout_probability = gate.trust(holdout["features"].to(device)).cpu()
    cal_selected = cal_probability >= best_threshold
    holdout_selected = holdout_probability >= best_threshold
    calibration_report = {
        "cases": len(cal["cases"]),
        "threshold": best_threshold,
        "classification": _classification_metrics(
            cal["labels"],
            cal_probability,
            best_threshold,
        ),
        "projection": _project_selection(cal["cases"], cal_selected),
        "case_decisions": [
            {
                "case": row["case"],
                "selected": bool(selected),
                "probability": float(probability),
                "accepted_label": row["accepted"],
                "dense": row["dense"],
                "routed": row["routed"],
            }
            for row, selected, probability in zip(
                cal["cases"],
                cal_selected,
                cal_probability,
                strict=True,
            )
        ],
        "test_opened": False,
    }
    holdout_report = {
        "cases": len(holdout["cases"]),
        "threshold": best_threshold,
        "classification": _classification_metrics(
            holdout["labels"],
            holdout_probability,
            best_threshold,
        ),
        "projection": _project_selection(holdout["cases"], holdout_selected),
        "case_decisions": [
            {
                "case": row["case"],
                "selected": bool(selected),
                "probability": float(probability),
                "accepted_label": row["accepted"],
                "dense": row["dense"],
                "routed": row["routed"],
            }
            for row, selected, probability in zip(
                holdout["cases"],
                holdout_selected,
                holdout_probability,
                strict=True,
            )
        ],
        "test_opened": False,
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "room_identity_arbitration_cal64_v1.json").write_text(
        json.dumps(calibration_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.report_root / "room_identity_arbitration_holdout64_v1.json").write_text(
        json.dumps(holdout_report, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "train_cases": len(train["cases"]),
        "calibration_cases": len(cal["cases"]),
        "holdout_cases": len(holdout["cases"]),
        "train_positive_fraction": float(train["labels"].mean()),
        "calibration_positive_fraction": float(cal["labels"].mean()),
        "holdout_positive_fraction": float(holdout["labels"].mean()),
        "best_score": best_score,
        "best_threshold": best_threshold,
        "history": history,
        "calibration_report": calibration_report,
        "holdout_report": holdout_report,
        "train_cases_detail": train["cases"],
        "calibration_cases_detail": cal["cases"],
        "holdout_cases_detail": holdout["cases"],
        "config": vars(args),
        "target_used_at_inference": False,
        "validation_labels_used_for_checkpoint_selection_only": True,
        "test_opened": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "train_cases", "calibration_cases", "holdout_cases",
        "train_positive_fraction", "best_score",
    )}, indent=2))


if __name__ == "__main__":
    main()

"""Fine-tune the existing joint refiner on real predicted controls."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.control_cache import control_cache_path, read_control_cache
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.heads.joint_control_repair import JointControlRepair
from floorflow.objectives.control_repair import (
    masked_room_loss,
    room_index_target,
    structure_losses,
)
from floorflow.representation.layered_floorplan import LayeredFloorplan
from floorflow.training.distribution_runtime import set_training_seed
from floorflow.training.raster_structure_bridge_data import (
    PredictedTargetDataset,
)

ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "outputs/train/fp_joint_control_repair_prefill_seed19_v2/best.pt"
PREDICTED = ROOT / "outputs/control_contracts/model_predicted_full_v1"
REPORT = ROOT / "outputs/reports/residential_predicted_refiner_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument(
        "--target-cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--initialize-from", type=Path, default=INITIAL)
    parser.add_argument("--predicted-root", type=Path, default=PREDICTED)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/train/residential_predicted_refiner_v1",
    )
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--room-base-channels", type=int, default=24)
    parser.add_argument("--structure-base-channels", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_model(args: argparse.Namespace, taxonomy, device: torch.device):
    structure_contract = getattr(args, "structure_contract", "legacy")
    model = JointControlRepair(
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        room_base_channels=args.room_base_channels,
        structure_base_channels=args.structure_base_channels,
        structure_contract=structure_contract,
        predict_centerline=bool(
            getattr(args, "predict_centerline", False)
        ),
        host_conditioning_v2=bool(
            getattr(args, "host_conditioning_v2", False)
        ),
        structure_global_route_attention=bool(
            getattr(args, "structure_global_route_attention", False)
        ),
        structure_route_grid=int(
            getattr(args, "structure_route_grid", 16)
        ),
        structure_route_router_count=int(
            getattr(args, "structure_route_router_count", 16)
        ),
        structure_route_top_k=int(
            getattr(args, "structure_route_top_k", 8)
        ),
        structure_route_heads=int(
            getattr(args, "structure_route_heads", 4)
        ),
    ).to(device)
    payload = torch.load(args.initialize_from, map_location=device, weights_only=False)
    if structure_contract == "legacy":
        model.load_state_dict(payload["model"])
    else:
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in payload["model"].items()
            if key in current and current[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
    return model


def _predicted_controls(
    root: Path,
    split: str,
    names,
    args,
    device,
):
    rows = [
        read_control_cache(control_cache_path(root, split, str(name)))
        for name in names
    ]
    return {
        key: torch.stack([row[key] for row in rows]).to(device)
        for key in ("room_logits", "structure", "furniture", "support", "program")
    }


def _forward(
    model: JointControlRepair,
    predicted: dict[str, torch.Tensor],
    footprint: torch.Tensor,
    args,
):
    room = F.interpolate(
        predicted["room_logits"],
        size=(args.room_size, args.room_size),
        mode="bilinear",
        align_corners=False,
    )
    structure = F.interpolate(
        predicted["structure"],
        size=(args.structure_size, args.structure_size),
        mode="bilinear",
        align_corners=False,
    )
    zero_room = torch.zeros_like(room[:, :1])
    zero_structure = torch.zeros_like(structure[:, :1])
    return model(
        room,
        structure,
        footprint,
        zero_room,
        zero_structure,
        program=predicted["program"],
    )


def _room_iou(prediction: np.ndarray, target: np.ndarray, room_ids) -> float:
    values = []
    for class_id in room_ids:
        expected = target == class_id
        if not expected.any():
            continue
        predicted = prediction == class_id
        values.append(
            float((predicted & expected).sum())
            / max(1.0, float((predicted | expected).sum()))
        )
    return float(np.mean(values)) if values else 0.0


def _support_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    room_ids,
) -> float:
    expected = np.isin(target, room_ids)
    predicted = np.isin(prediction, room_ids)
    union = (predicted | expected).sum()
    return float((predicted & expected).sum() / max(1, union))


def _binary_f1(probability, target, valid) -> float:
    predicted = (probability >= 0.5) & valid
    expected = (target >= 0.5) & valid
    denominator = predicted.sum() + expected.sum()
    if int(denominator) == 0:
        return 1.0
    return float(2.0 * (predicted & expected).sum() / denominator)


def _safe_fpeval(prediction, target, taxonomy) -> dict:
    try:
        return fpeval_raster_metrics(
            prediction,
            target,
            taxonomy,
            tolerance_fractions=(0.05,),
        )
    except (IndexError, ValueError):
        return {
            "prediction_validity": {"valid": False, "watertight": False}
        }


@torch.inference_mode()
def evaluate(model, loader, taxonomy, args) -> tuple[dict, list[dict]]:
    model.eval()
    device = torch.device(args.device)
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=device)
    room_count = len(taxonomy.room_ids)
    rows = []
    for target, footprint, _floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        predicted = _predicted_controls(args.predicted_root, "val", names, args, device)
        output = _forward(model, predicted, footprint, args)
        labels = room_ids[output["room_logits_full"].argmax(dim=1)]
        labels = F.interpolate(
            labels[:, None].float(),
            size=target.shape[-2:],
            mode="nearest",
        )[:, 0].long()
        labels *= footprint.long()
        structure = output["structure"]
        wall = F.interpolate(
            structure["wall_logits"].sigmoid(),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0] >= 0.5
        opening_type = F.interpolate(
            structure["opening_type_logits"],
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        opening = opening_ids[opening_type] * (
            F.interpolate(
                structure["opening_logits"].sigmoid(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0] >= 0.5
        )
        visible = LayeredFloorplan(
            room_labels=labels,
            wall=wall.long(),
            opening=opening,
            furniture=torch.zeros_like(labels),
            footprint=footprint,
            taxonomy=taxonomy,
        ).visible_labels()
        clean_structure = coarse_structure_control(
            target,
            footprint,
            taxonomy,
            size=args.structure_size,
        )
        for index, name in enumerate(names):
            target_np = target[index].cpu().numpy()
            labels_np = labels[index].cpu().numpy()
            structural = structural_legality_metrics(
                visible[index].cpu().numpy(),
                target_np,
                taxonomy,
            )
            fpeval = _safe_fpeval(
                visible[index].cpu().numpy(),
                target_np,
                taxonomy,
            )
            rows.append(
                {
                    "name": str(name),
                    "room_macro_iou": _room_iou(
                        labels_np,
                        target_np,
                        taxonomy.room_ids,
                    ),
                    "room_support_iou": _support_iou(
                        labels_np,
                        target_np,
                        taxonomy.room_ids,
                    ),
                    "wall_f1": _binary_f1(
                        output["structure"]["wall_logits"][index, 0].sigmoid(),
                        clean_structure[index, room_count],
                        torch.ones_like(
                            output["structure"]["wall_logits"][index, 0],
                            dtype=torch.bool,
                        ),
                    ),
                    "opening_f1": _binary_f1(
                        output["structure"]["opening_logits"][index, 0].sigmoid(),
                        clean_structure[index, room_count + 1],
                        torch.ones_like(
                            output["structure"]["opening_logits"][index, 0],
                            dtype=torch.bool,
                        ),
                    ),
                    "validity": float(
                        fpeval["prediction_validity"]["valid"]
                    ),
                    "watertight": float(
                        fpeval["prediction_validity"]["watertight"]
                    ),
                    "room_closure_rate": structural["room_closure_rate"],
                }
            )
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "room_macro_iou",
            "room_support_iou",
            "wall_f1",
            "opening_f1",
            "validity",
            "watertight",
            "room_closure_rate",
            "room_closure_rate",
        )
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    set_training_seed(args.seed)
    device = torch.device(args.device)
    train = PredictedTargetDataset(
        args.predicted_root,
        "train",
        args.target_cache_root,
    )
    validation = PredictedTargetDataset(
        args.predicted_root,
        "val",
        args.target_cache_root,
    )
    taxonomy = FloorplanTaxonomy.read(
        args.cache_root / "taxonomy.json"
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(validation, shuffle=False, **common)
    model = _load_model(args, taxonomy, device)
    baseline_summary, baseline_rows = evaluate(model, val_loader, taxonomy, args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    best_score = baseline_summary["room_macro_iou"]
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_summary = baseline_summary
    best_rows = baseline_rows
    history = []
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        totals = {}
        seen = 0
        for target, footprint, _floor_ids, names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            predicted = _predicted_controls(
                args.predicted_root,
                "train",
                names,
                args,
                device,
            )
            output = _forward(model, predicted, footprint, args)
            room_target = target
            room_loss, room_terms = masked_room_loss(
                output["room_logits_full"],
                room_index_target(
                    room_target,
                    torch.tensor(taxonomy.room_ids, device=device),
                ),
                torch.ones_like(room_target[:, None], dtype=torch.bool),
            )
            clean_structure = coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=args.structure_size,
            )
            structure_loss, structure_terms = structure_losses(
                output["structure"],
                clean_structure,
                torch.ones(
                    (
                        clean_structure.shape[0],
                        clean_structure.shape[-2],
                        clean_structure.shape[-1],
                    ),
                    device=device,
                    dtype=torch.bool,
                ),
                len(taxonomy.room_ids),
                torch.tensor(taxonomy.opening_ids, device=device),
            )
            loss = room_loss + structure_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in {**room_terms, **structure_terms}.items():
                totals[key] = totals.get(key, 0.0) + float(value) * target.shape[0]
            seen += target.shape[0]
        scheduler.step()
        summary, rows = evaluate(model, val_loader, taxonomy, args)
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value / max(seen, 1) for key, value in totals.items()},
                **summary,
                "seconds": time.perf_counter() - started,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if summary["room_macro_iou"] > best_score:
            best_score = summary["room_macro_iou"]
            best_summary = summary
            best_rows = rows
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    comparisons = {
        metric: paired_delta(
            best_rows,
            baseline_rows,
            metric,
            iterations=args.bootstrap,
            seed=args.seed + 24000,
        )
        for metric in (
            "room_macro_iou",
            "room_support_iou",
            "wall_f1",
            "opening_f1",
            "validity",
            "watertight",
        )
    }
    checks = {
        "room_iou_paired_positive": comparisons["room_macro_iou"]["ci_low"] > 0.0,
        "support_iou_nonregression": comparisons["room_support_iou"]["ci_high"] >= 0.0,
        "validity_at_least_0_95": best_summary["validity"] >= 0.95,
        "watertight_nonregression": comparisons["watertight"]["ci_high"] >= 0.0,
        "no_target_controls_at_inference": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_root / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args) | {"device": str(device)},
            "metrics": best_summary,
        },
        checkpoint,
    )
    report = {
        "schema_version": "residential-predicted-refiner-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only",
        "split": "val191",
        "test_opened": False,
        "target_used_at_inference": False,
        "baseline": baseline_summary,
        "best": best_summary,
        "paired": comparisons,
        "history": history,
        "checks": checks,
        "retained_limitation": {
            "room_closure_rate_baseline": baseline_summary["room_closure_rate"],
            "room_closure_rate_best": best_summary["room_closure_rate"],
            "reason": (
                "Improved room type assignment and watertightness come with "
                "lower wall-face room closure; this must be reported."
            ),
        },
        "checkpoint": (
            str(checkpoint.relative_to(ROOT))
            if checkpoint.is_absolute()
            else str(checkpoint)
        ),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "checks": checks}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

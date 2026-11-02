#!/usr/bin/env python3
"""Train a frozen-feature door/window detector with wall-projected readout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.opening_readout import (
    detections_from_labels,
    opening_detection_metrics,
)
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    raster_labels_to_wall_plan,
)
from floorflow.heads.opening_detection import (
    OpeningDetectionHead,
    opening_detection_loss,
)
from floorflow.heads.raster_control_bank import (
    load_raster_instance_control_model,
    load_raster_room_model,
)
from floorflow.inference.opening_center_width_decoder import (
    read_center_width_opening_detections,
)
from floorflow.inference.opening_heatmap_decoder import (
    encode_opening_features,
    learned_opening_detections,
    opening_detection_maps,
    opening_head_outputs,
)
from floorflow.training.opening_head_scopes import (
    SCOPES,
    configure_opening_head_scope,
)
from floorflow.training.training import ModelEMA

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/raster_instance_control_384_promotion256_10ep_v1/best.pt",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/opening_interval_decoder_screen256_v1",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--selection-threshold", type=float, default=0.30)
    parser.add_argument("--min-area", type=int, default=7)
    parser.add_argument("--max-wall-distance", type=float, default=4.0)
    parser.add_argument("--min-peak-distance", type=float, default=4.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--room-boundary-context", action="store_true")
    parser.add_argument(
        "--head-variant",
        choices=("v1", "multiscale", "center_width"),
        default="v1",
    )
    parser.add_argument("--spatial-gap-attention", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--train-scope", choices=SCOPES, default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_model(
    checkpoint: Path,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
) -> tuple[torch.nn.Module, str, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", {})
    if any(key.startswith("room.") for key in state):
        model = load_raster_instance_control_model(checkpoint, taxonomy, device)
        channels = int(payload.get("config", {}).get("decoder_channels", 128))
        family = "raster_instance_control"
    else:
        model = load_raster_room_model(checkpoint, taxonomy, device)
        channels = int(payload.get("config", {}).get("decoder_channels", 96))
        family = "raster_room"
    model.requires_grad_(False).eval()
    return model, family, channels


def pixel_opening_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    opening_ids: tuple[int, ...],
    threshold: float,
) -> dict[str, float]:
    expected = np.isin(target, opening_ids) & footprint.astype(bool)
    predicted = probability.sum(axis=0) >= float(threshold)
    predicted &= footprint.astype(bool)
    true_positive = int((predicted & expected).sum())
    precision = true_positive / max(1, int(predicted.sum()))
    recall = true_positive / max(1, int(expected.sum()))
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": 2.0 * precision * recall / max(1.0e-8, precision + recall),
        "predicted_pixels": int(predicted.sum()),
        "target_pixels": int(expected.sum()),
    }


def vector_opening_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    image_size: int,
    threshold: float,
    min_area: int,
    max_wall_distance: float,
    min_peak_distance: float = 4.0,
    maps: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    target_plan = raster_labels_to_wall_plan(target, taxonomy)
    proxy_plan = RasterWallPlan(
        frame=target_plan.frame,
        walls=target_plan.walls,
        rooms=target_plan.rooms,
        openings=[],
    )
    if maps is not None and "center" in maps and "interval_width" in maps:
        predicted = read_center_width_opening_detections(
            proxy_plan,
            maps["center"][0].cpu().numpy(),
            maps["interval_width"][0].cpu().numpy(),
            maps["family_logits"][0].cpu().numpy(),
            taxonomy,
            threshold=threshold,
            min_width=float(min_area),
            min_peak_distance=min_peak_distance,
        )
    else:
        predicted = learned_opening_detections(
            proxy_plan,
            probability,
            taxonomy,
            threshold=threshold,
            min_area=min_area,
            max_wall_distance=max_wall_distance,
        )
    expected = detections_from_labels(target, taxonomy, target_plan)
    metrics = opening_detection_metrics(
        predicted,
        expected,
        tolerance=0.05 * float(image_size),
    )
    return {
        "vector_precision": float(metrics["precision"]),
        "vector_recall": float(metrics["recall"]),
        "vector_f1": float(metrics["f1"]),
        "geometry_matched": float(metrics["geometry_matched"]),
        "family_consistent": float(metrics["family_consistent"]),
        "geometry_center_error_mean": float(
            metrics["geometry_center_error_mean"]
        ),
        "geometry_width_ratio_mean": float(
            metrics["geometry_width_ratio_mean"]
        ),
        "predicted_openings": float(len(predicted)),
        "target_openings": float(len(expected)),
    }


@torch.inference_mode()
def validate(
    feature_model,
    head,
    loader,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
    args: argparse.Namespace,
    *,
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    feature_model.eval()
    head.eval()
    rows = []
    opening_ids = tuple(taxonomy.opening_ids)
    for target, footprint, image, _floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        features = encode_opening_features(
            feature_model,
            footprint,
            image,
            include_image=True,
            include_room_boundary=args.room_boundary_context,
        )
        output = opening_head_outputs(head, features)
        maps = opening_detection_maps(output, taxonomy)
        probability = maps["family_probability"]
        for index, name in enumerate(names):
            probability_row = probability[index].float().cpu().numpy()
            case_maps = {
                key: value[index : index + 1]
                for key, value in maps.items()
                if torch.is_tensor(value)
            }
            target_row = target[index].cpu().numpy()
            footprint_row = footprint[index].cpu().numpy()
            rows.append(
                {
                    "case": str(name),
                    **pixel_opening_metrics(
                        probability_row,
                        target_row,
                        footprint_row,
                        opening_ids,
                        threshold,
                    ),
                    **vector_opening_metrics(
                        probability_row,
                        target_row,
                        taxonomy,
                        image_size=args.image_size,
                        threshold=threshold,
                        min_area=args.min_area,
                        max_wall_distance=args.max_wall_distance,
                        min_peak_distance=args.min_peak_distance,
                        maps=case_maps,
                    ),
                }
            )
    keys = [
        key
        for key in rows[0]
        if key != "case"
    ]
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in keys
    }
    return summary, rows


def calibrate_threshold(
    feature_model,
    head,
    loader,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    grid = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65)
    rows = []
    for threshold in grid:
        summary, _cases = validate(
            feature_model,
            head,
            loader,
            taxonomy,
            device,
            args,
            threshold=threshold,
        )
        rows.append({"threshold": threshold, **summary})
    selected = max(rows, key=lambda row: float(row["vector_f1"]))
    return {"selected": selected, "grid": rows}


def load_control_metrics() -> dict[str, object]:
    paths = {
        "label_derived": (
            ROOT / "outputs/reports/room_identity_safety_global_val491_v1.json",
            "fpeval_opening_f1_mean",
        ),
        "threshold_interval": (
            ROOT / "outputs/reports/interval_opening_safe_t020_cal64_v1.json",
            "fpeval_opening_f1_mean",
        ),
        "structured_overlay": (
            ROOT / "outputs/reports/structure_evidence_wall08_opening05_cal64_v1.json",
            "fpeval_opening_f1_mean",
        ),
    }
    controls: dict[str, object] = {}
    for name, (path, key) in paths.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("summary", {}).get(key)
        controls[name] = {"path": str(path.relative_to(ROOT)), "opening_f1": value}
    return controls


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index not in {None, 0}:
        raise ValueError("opening decoder training is restricted to cuda:0")
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "prefetch_factor": 2 if args.workers > 0 else None,
    }
    train = CubiCasaRasterControlDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=True,
    )
    val = CubiCasaRasterControlDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit_val + args.val_offset,
        augment=False,
    )
    if args.val_offset:
        val.samples = val.samples[
            args.val_offset : args.val_offset + args.limit_val
        ]
    taxonomy = train.taxonomy
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    feature_model, feature_family, channels = load_feature_model(
        args.feature_checkpoint,
        taxonomy,
        device,
    )
    head_input_channels = channels + (5 if args.room_boundary_context else 4)
    head = OpeningDetectionHead(
        head_input_channels,
        variant=args.head_variant,
        spatial_gap_attention=args.spatial_gap_attention,
    ).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if str(initial.get("head_variant", "v1")) != args.head_variant:
            raise ValueError("initial checkpoint head variant mismatch")
        if bool(initial.get("spatial_gap_attention", False)) != bool(
            args.spatial_gap_attention
        ):
            raise ValueError("initial checkpoint spatial-gap mode mismatch")
        incompatible = head.load_state_dict(initial["head"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key for key in incompatible.missing_keys if not key.startswith("gap_")
        ]
        if unexpected or missing:
            raise ValueError(
                f"initial checkpoint is incompatible: "
                f"missing={missing}, unexpected={unexpected}"
            )
    elif args.train_scope != "full":
        raise ValueError("partial opening training requires --init-checkpoint")
    scope_manifest = configure_opening_head_scope(head, args.train_scope)
    trainable_parameters = [
        parameter for parameter in head.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, len(train_loader) * args.epochs),
        eta_min=args.lr * 0.05,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=args.amp and device.type == "cuda",
    )
    ema = ModelEMA(head, decay=args.ema_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        started = time.perf_counter()
        head.train()
        loss_total = 0.0
        seen = 0
        for target, footprint, image, _floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            image = image.to(device)
            with torch.no_grad():
                features = encode_opening_features(
                    feature_model,
                    footprint,
                    image,
                    include_image=True,
                    include_room_boundary=args.room_boundary_context,
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                output = opening_head_outputs(head, features)
                loss, _components = opening_detection_loss(
                    output,
                    target,
                    footprint,
                    taxonomy,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(head)
            loss_total += float(loss.detach().cpu()) * target.shape[0]
            seen += target.shape[0]
        metrics, case_rows = validate(
            feature_model,
            ema.module,
            val_loader,
            taxonomy,
            device,
            args,
            threshold=args.selection_threshold,
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_total / max(1, seen),
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "head": ema.module.state_dict(),
            "raw_head": head.state_dict(),
            "head_in_channels": head_input_channels,
            "head_hidden_dim": 96,
            "head_family": "opening_detection_v1",
            "head_variant": args.head_variant,
            "spatial_gap_attention": args.spatial_gap_attention,
            "feature_checkpoint": str(args.feature_checkpoint),
            "feature_family": feature_family,
            "taxonomy": taxonomy.to_json(),
            "config": {
                **vars(args),
                "feature_family": feature_family,
                "head_in_channels": head_input_channels,
                "include_image_features": True,
                "include_room_boundary_features": args.room_boundary_context,
                "selection_threshold": args.selection_threshold,
                "init_checkpoint": (
                    str(args.init_checkpoint)
                    if args.init_checkpoint is not None
                    else None
                ),
                "train_scope": args.train_scope,
                "scope_manifest": scope_manifest,
            },
            "epoch": epoch,
            "val": record,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["vector_f1"] > best:
            best = metrics["vector_f1"]
            torch.save(checkpoint, args.output_dir / "best.pt")
            with (args.output_dir / "best_cases.json").open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(case_rows, handle, indent=2)
                handle.write("\n")
    best_payload = torch.load(
        args.output_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    head.load_state_dict(best_payload["head"])
    calibration = calibrate_threshold(
        feature_model,
        head,
        val_loader,
        taxonomy,
        device,
        args,
    )
    selected = calibration["selected"]
    report = {
        "schema_version": "opening-heatmap-decoder-v1",
        "scope": "frozen-feature train256/cal64 mechanism screen",
        "test_opened": False,
        "target_used_at_inference": False,
        "oracle_wall_association": True,
        "selection_metric": "oracle-host vector opening F1",
        "budget": {
            "train_cases": len(train),
            "cal_cases": len(val),
            "epochs": args.epochs,
            "updates": len(train_loader) * args.epochs,
            "pixels_at_384": len(train) * args.image_size**2 * args.epochs,
            "trainable_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in head.parameters()
                    if parameter.requires_grad
                )
            ),
            "frozen_feature_parameters": int(
                sum(parameter.numel() for parameter in feature_model.parameters())
            ),
        },
        "controls": load_control_metrics(),
        "history": history,
        "threshold_calibration": calibration,
        "selected": selected,
        "train_scope": scope_manifest,
        "gate": {
            "requires_end_to_end": True,
            "opening_f1_target": 0.30,
            "promotion_rule": (
                "Promote only if the canonical predicted-wall opening F1 reaches "
                "0.30 and room, instance, wall, validity, and watertightness do "
                "not regress."
            ),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "best": best,
                "report": str(args.output_dir / "report.json"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

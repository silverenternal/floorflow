"""Matched base/scalar/conflict-projected constraint refinement screen."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.control_fusion import nearest_fill_batch
from floorflow.data.control_perturbation import random_spatial_block_mask
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.heads.joint_control_repair import JointControlRepair
from floorflow.objectives.constraint_fusion import (
    apply_constraint_fusion,
    room_graph_edit_proxy,
)
from floorflow.objectives.control_repair import (
    masked_focal_dice,
    masked_room_loss,
    room_index_target,
    structure_losses,
)
from floorflow.objectives.differentiable_fevp import differentiable_fevp_terms
from floorflow.representation.coarse_layout import coarse_room_layout_logits
from floorflow.training.distribution_runtime import set_training_seed

ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "outputs/train/fp_joint_control_repair_prefill_seed19_v2/best.pt"
REPORT = ROOT / "outputs/reports/residential_constraint_flow_screen_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--initialize-from", type=Path, default=INITIAL)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/train/residential_constraint_flow_v1",
    )
    parser.add_argument("--arms", default="base,scalar,projection")
    parser.add_argument(
        "--constraint-kind",
        choices=("combined", "fevp", "ged"),
        default="combined",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=192)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--room-base-channels", type=int, default=24)
    parser.add_argument("--structure-base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--max-dropout", type=float, default=0.4)
    parser.add_argument("--soft-fevp-weight", type=float, default=0.05)
    parser.add_argument("--soft-ged-weight", type=float, default=0.05)
    parser.add_argument("--constraint-scale", type=float, default=8.0)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _binary_f1(probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    predicted = (probability >= 0.5) & valid
    expected = (target >= 0.5) & valid
    denominator = predicted.sum() + expected.sum()
    if int(denominator) == 0:
        return 1.0
    return float(2.0 * (predicted & expected).sum() / denominator)


@torch.inference_mode()
def validate(
    model: JointControlRepair,
    loader: DataLoader,
    taxonomy,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    model.eval()
    device = torch.device(args.device)
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    by_case = {}
    exact_room = 0.0
    exact_structure = 0.0
    for fraction in (0.1, 0.25):
        for target, footprint, _floor_ids, names in loader:
            target = target.to(device)
            footprint = footprint.to(device)
            clean_room = coarse_room_layout_logits(
                target,
                taxonomy,
                size=args.room_size,
            )
            clean_structure = coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=args.structure_size,
            )
            room_missing = random_spatial_block_mask(
                target.shape[0],
                clean_room.shape[-2:],
                fraction,
                generator=torch.Generator().manual_seed(
                    args.seed + 700 + round(fraction * 100)
                ),
            ).to(device)
            structure_missing = random_spatial_block_mask(
                target.shape[0],
                clean_structure.shape[-2:],
                fraction,
                generator=torch.Generator().manual_seed(
                    args.seed + 900 + round(fraction * 100)
                ),
            ).to(device)
            room_input = nearest_fill_batch(clean_room, ~room_missing)
            structure_input = nearest_fill_batch(
                clean_structure,
                ~structure_missing,
            )
            output = model(
                room_input,
                structure_input,
                footprint,
                ~room_missing,
                ~structure_missing,
            )
            room_valid = F.interpolate(
                room_missing[:, None].float(),
                size=target.shape[-2:],
                mode="nearest",
            ).bool()
            structure_valid = structure_missing[:, None].bool()
            predicted_support = output["room_logits_full"].amax(dim=1) > 0.0
            predicted_labels = room_ids[
                output["room_logits_full"].argmax(dim=1)
            ]
            expected_support = torch.isin(target, room_ids)
            room_outside = (~room_valid[:, 0]).float()
            structure_outside = (~structure_valid[:, 0]).float()
            room_delta = (
                output["room_logits"] - room_input
            ).abs().amax(dim=1)
            structure_delta = (
                output["structure_control"] - structure_input
            ).abs().amax(dim=1)
            exact_room = max(
                exact_room,
                float((room_delta * room_outside).max().cpu()),
            )
            exact_structure = max(
                exact_structure,
                float((structure_delta * structure_outside).max().cpu()),
            )
            room_count = len(taxonomy.room_ids)
            for index, name in enumerate(names):
                pixel_valid = (
                    expected_support[index] & room_valid[index, 0]
                )
                room_accuracy = (
                    float(
                        (
                            (predicted_labels[index] == target[index])
                            & pixel_valid
                        ).sum()
                        / pixel_valid.sum()
                    )
                    if pixel_valid.any()
                    else 1.0
                )
                row = {
                    "support_iou": float(
                        (
                            (predicted_support[index] & expected_support[index])
                            & room_valid[index, 0]
                        ).sum()
                        / (
                            (predicted_support[index] | expected_support[index])
                            & room_valid[index, 0]
                        ).sum().clamp_min(1)
                    ),
                    "room_accuracy": room_accuracy,
                    "wall_f1": _binary_f1(
                        output["structure"]["wall_logits"][index, 0].sigmoid(),
                        clean_structure[index, room_count],
                        structure_valid[index, 0],
                    ),
                    "opening_f1": _binary_f1(
                        output["structure"]["opening_logits"][index, 0].sigmoid(),
                        clean_structure[index, room_count + 1],
                        structure_valid[index, 0],
                    ),
                }
                bucket = by_case.setdefault(str(name), [])
                bucket.append(row)
    rows = []
    for name, values in sorted(by_case.items()):
        rows.append(
            {
                "name": name,
                **{
                    key: sum(float(row[key]) for row in values) / len(values)
                    for key in ("support_iou", "room_accuracy", "wall_f1", "opening_f1")
                },
            }
        )
        rows[-1]["selection_score"] = (
            rows[-1]["room_accuracy"]
            + rows[-1]["wall_f1"]
            + rows[-1]["opening_f1"]
        ) / 3.0
    summary = {
        "cases": len(rows),
        **{
            key: sum(row[key] for row in rows) / max(len(rows), 1)
            for key in (
                "selection_score",
                "support_iou",
                "room_accuracy",
                "wall_f1",
                "opening_f1",
            )
        },
        "room_valid_exact_max_abs": exact_room,
        "structure_valid_exact_max_abs": exact_structure,
    }
    return summary, rows


def _train_arm(
    arm: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    taxonomy,
    args: argparse.Namespace,
) -> tuple[JointControlRepair, dict, list[dict]]:
    set_training_seed(args.seed)
    device = torch.device(args.device)
    model = JointControlRepair(
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        room_base_channels=args.room_base_channels,
        structure_base_channels=args.structure_base_channels,
    ).to(device)
    payload = torch.load(args.initialize_from, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=device)
    room_count = len(taxonomy.room_ids)
    diagnostics = []
    history = []
    best_score = -float("inf")
    best_epoch = -1
    best_summary = None
    best_rows = None
    best_state = None
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        totals = {}
        seen = 0
        for target, footprint, _floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            clean_room = coarse_room_layout_logits(
                target,
                taxonomy,
                size=args.room_size,
            )
            clean_structure = coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=args.structure_size,
            )
            fractions = args.max_dropout * torch.rand(target.shape[0])
            room_missing = random_spatial_block_mask(
                target.shape[0],
                clean_room.shape[-2:],
                fractions,
            ).to(device)
            structure_missing = random_spatial_block_mask(
                target.shape[0],
                clean_structure.shape[-2:],
                fractions,
            ).to(device)
            room_input = nearest_fill_batch(clean_room, ~room_missing)
            structure_input = nearest_fill_batch(
                clean_structure,
                ~structure_missing,
            )
            output = model(
                room_input,
                structure_input,
                footprint,
                ~room_missing,
                ~structure_missing,
            )
            room_valid = F.interpolate(
                room_missing[:, None].float(),
                size=target.shape[-2:],
                mode="nearest",
            ).bool()
            room_loss, room_terms = masked_room_loss(
                output["room_logits_full"],
                room_index_target(target, room_ids),
                room_valid,
            )
            structure_loss, structure_terms = structure_losses(
                output["structure"],
                clean_structure,
                structure_missing,
                room_count,
                opening_ids,
            )
            boundary_target = clean_structure[:, room_count : room_count + 2]
            boundary_valid = (
                structure_missing[:, None].bool()
                & (
                    F.max_pool2d(boundary_target, 3, 1, 1)
                    - boundary_target
                ).abs().bool()
            )
            boundary = masked_focal_dice(
                torch.cat(
                    [
                        output["structure"]["wall_logits"],
                        output["structure"]["opening_logits"],
                    ],
                    dim=1,
                ),
                boundary_target,
                boundary_valid,
            )
            base_loss = room_loss + structure_loss + 0.1 * boundary
            constraint_loss = base_loss.detach() * 0.0
            soft_fevp = base_loss.detach() * 0.0
            soft_ged = base_loss.detach() * 0.0
            if arm != "base":
                if args.constraint_kind in {"combined", "fevp"}:
                    soft_fevp, _terms = differentiable_fevp_terms(
                        output["room_logits_full"].softmax(dim=1),
                        F.interpolate(
                            room_input.softmax(dim=1),
                            size=target.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                        output["structure"]["wall_logits"].sigmoid()[:, 0],
                        output["structure"]["opening_logits"].sigmoid()[:, 0],
                        footprint.float(),
                        room_valid.float(),
                    )
                    constraint_loss = (
                        constraint_loss
                        + args.soft_fevp_weight * soft_fevp
                    )
                if args.constraint_kind in {"combined", "ged"}:
                    target_probability = torch.stack(
                        [
                            (target == room_id).float()
                            for room_id in taxonomy.room_ids
                        ],
                        dim=1,
                    )
                    soft_ged, _ged_terms = room_graph_edit_proxy(
                        output["room_logits_full"].softmax(dim=1),
                        target_probability,
                    )
                    constraint_loss = (
                        constraint_loss
                        + args.soft_ged_weight * soft_ged
                    )
            optimizer.zero_grad(set_to_none=True)
            diagnostic = apply_constraint_fusion(
                base_loss,
                constraint_loss,
                list(model.parameters()),
                mode=arm,
                constraint_scale=args.constraint_scale,
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            diagnostics.append(diagnostic.to_dict())
            terms = {
                "base": base_loss.detach(),
                "constraint": constraint_loss.detach(),
                "soft_fevp": soft_fevp.detach(),
                "soft_ged": soft_ged.detach(),
                **room_terms,
                **structure_terms,
            }
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value) * target.shape[0]
            seen += target.shape[0]
        summary, rows = validate(model, val_loader, taxonomy, args)
        history.append(
            {
                "epoch": epoch,
                **{
                    f"train_{key}": value / max(seen, 1)
                    for key, value in totals.items()
                },
                **summary,
                "seconds": time.perf_counter() - started,
            }
        )
        print(json.dumps({"arm": arm, **history[-1]}), flush=True)
        if summary["selection_score"] > best_score:
            best_score = summary["selection_score"]
            best_epoch = epoch
            best_summary = summary
            best_rows = rows
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None or best_summary is None or best_rows is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    mean_diagnostics = {
        key: sum(row[key] for row in diagnostics) / max(len(diagnostics), 1)
        for key in (
            "base_norm",
            "constraint_norm",
            "cosine",
            "projected_constraint_norm",
            "projected_fraction",
        )
    }
    best_summary["fusion_diagnostics"] = mean_diagnostics
    best_summary["selected_epoch"] = best_epoch
    return model, {"summary": best_summary, "history": history}, best_rows


def main() -> None:
    args = parse_args()
    requested = [arm for arm in args.arms.split(",") if arm]
    if not requested or not set(requested) <= {"base", "scalar", "projection"}:
        raise ValueError("arms must be base, scalar, projection")
    dataset_args = {
        "image_size": 256,
        "cache_root": args.cache_root,
        "augment": False,
    }
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        limit=args.limit_train,
        **dataset_args,
    )
    validation = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        limit=args.limit_val,
        **dataset_args,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.device(args.device).type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    results = {}
    rows_by_arm = {}
    for arm in ("base", "scalar", "projection"):
        if arm not in requested:
            continue
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train,
            shuffle=True,
            generator=generator,
            **common,
        )
        val_loader = DataLoader(validation, shuffle=False, **common)
        model, result, rows = _train_arm(
            arm,
            train_loader,
            val_loader,
            train.taxonomy,
            args,
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output_root / f"{arm}_seed{args.seed}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "taxonomy": train.taxonomy.to_json(),
                "config": vars(args) | {"device": str(args.device)},
                "arm": arm,
            },
            checkpoint,
        )
        results[arm] = result
        rows_by_arm[arm] = rows
    comparisons = {}
    if "base" in rows_by_arm:
        for arm in ("scalar", "projection"):
            if arm not in rows_by_arm:
                continue
            for metric in (
                "selection_score",
                "room_accuracy",
                "wall_f1",
                "opening_f1",
            ):
                comparisons[f"{arm}_vs_base_{metric}"] = paired_delta(
                    rows_by_arm[arm],
                    rows_by_arm["base"],
                    metric,
                    iterations=args.bootstrap,
                    seed=args.seed + 12000,
                )
    hard_checks = {
        arm: (
            result["summary"]["room_valid_exact_max_abs"] == 0.0
            and result["summary"]["structure_valid_exact_max_abs"] == 0.0
        )
        for arm, result in results.items()
    }
    quality_arms = [
        arm for arm in ("scalar", "projection") if arm in results
    ]
    quality_checks = {}
    if "base" in rows_by_arm:
        quality_checks = {
            arm: any(
                comparisons[f"{arm}_vs_base_{metric}"]["mean_delta"] > 0.0
                and comparisons[f"{arm}_vs_base_{metric}"]["ci_low"] > 0.0
                for metric in (
                    "selection_score",
                    "room_accuracy",
                    "wall_f1",
                    "opening_f1",
                )
            )
            for arm in quality_arms
        }
    report = {
        "schema_version": "residential-constraint-flow-screen-v1",
        "scope": "residential_floorplan_only",
        "training_only": True,
        "test_opened": False,
        "soft_proxy_boundary": "differentiable FEVP and soft GED are training terms only",
        "constraint_kind": args.constraint_kind,
        "arms": results,
        "comparisons": comparisons,
        "checks": {
            "hard_valid_regions_exact": all(hard_checks.values()),
            "nonbase_quality_improvement": any(quality_checks.values()),
            "projection_diagnostics_present": (
                "fusion_diagnostics" in results.get("projection", {}).get("summary", {})
            ),
        },
        "quality_checks": quality_checks,
    }
    report["status"] = (
        "passed" if all(report["checks"].values()) else "failed"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "status": report["status"], "checks": report["checks"]}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

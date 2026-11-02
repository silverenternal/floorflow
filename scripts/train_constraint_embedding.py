"""Train the self-supervised constraint state encoder on CubiCasa controls."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from floorflow.evaluation.constraint_calibration import (
    apply_temperature,
    binary_auroc,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    select_threshold,
)
from floorflow.model.constraint_state_encoder import (
    ConstraintStateEncoder,
    d4_transform_graph,
)
from floorflow.objectives.constraint_contrastive import (
    energy_ranking_loss,
    multi_label_violation_loss,
    multi_positive_infonce,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/residential_cagn_features_v1",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "outputs/residential_cagn_embedding_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/residential_constraint_embedding_v1.json",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_features(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["node_features"] = payload["node_features"].float()
    payload["adjacency"] = payload["adjacency"].float()
    payload["violation_labels"] = payload["violation_labels"].float()
    payload["legal"] = payload["legal"].bool()
    payload["node_mask"] = payload["node_mask"].bool()
    return payload


def _subset(features: dict, mask: torch.Tensor) -> dict:
    tensor_keys = (
        "node_features",
        "adjacency",
        "global_features",
        "action_features",
        "hard_mask",
        "constraint_diagnostics",
        "node_mask",
        "operation_ids",
        "violation_labels",
        "legal",
        "group_index",
        "room_count",
    )
    return {
        **{key: features[key][mask] for key in tensor_keys},
        "family_ids": features["family_ids"],
        "violation_families": features["violation_families"],
    }


def _move(features: dict, indices: torch.Tensor, device: torch.device) -> dict:
    sample_keys = (
        "node_features",
        "adjacency",
        "global_features",
        "action_features",
        "hard_mask",
        "constraint_diagnostics",
        "node_mask",
        "operation_ids",
        "violation_labels",
        "legal",
        "group_index",
        "room_count",
    )
    return {
        **{
            key: features[key][indices].to(device)
            for key in sample_keys
        },
        "family_ids": features["family_ids"].to(device),
    }


def _augment(features: dict, generator: torch.Generator) -> dict:
    batch, nodes, _ = features["node_features"].shape
    transformations = torch.randint(0, 8, (batch,), generator=generator)
    transformations = transformations.to(features["node_features"].device)
    variants = torch.stack(
        [
            d4_transform_graph(
                features["node_features"],
                features["adjacency"],
                transform,
            )[0]
            for transform in range(8)
        ],
        dim=0,
    )
    gather_index = transformations[None, :, None, None].expand(
        1,
        batch,
        nodes,
        features["node_features"].shape[-1],
    )
    rotated_nodes = variants.gather(0, gather_index)[0]
    device_generator = torch.Generator(device=features["node_features"].device)
    device_generator.manual_seed(generator.initial_seed() + 17)
    random_scores = torch.rand(
        batch,
        nodes,
        generator=device_generator,
        device=features["node_features"].device,
    )
    permutations = torch.argsort(random_scores, dim=1)
    permuted_nodes = torch.gather(
        features["node_features"],
        1,
        permutations[..., None].expand_as(features["node_features"]),
    )
    permuted_adjacency = torch.gather(
        features["adjacency"],
        1,
        permutations[:, :, None].expand(
            batch,
            nodes,
            nodes,
        ),
    )
    permuted_adjacency = torch.gather(
        permuted_adjacency,
        2,
        permutations[:, None, :].expand(
            batch,
            nodes,
            nodes,
        ),
    )
    permuted_family = torch.gather(
        features["family_ids"][None].expand(batch, nodes),
        1,
        permutations,
    )
    permuted_mask = torch.gather(
        features["node_mask"],
        1,
        permutations,
    )
    return {
        "rotated": {
            **features,
            "node_features": rotated_nodes,
            "family_ids": features["family_ids"][None].expand(batch, -1),
        },
        "permuted": {
            **features,
            "node_features": permuted_nodes,
            "adjacency": permuted_adjacency,
            "family_ids": permuted_family,
            "node_mask": permuted_mask,
        },
    }


def _model_input(features: dict) -> tuple[torch.Tensor, ...]:
    family = features["family_ids"]
    if family.ndim == 1:
        family = family[None].expand(features["node_features"].shape[0], -1)
    return (
        features["node_features"],
        features["adjacency"],
        family,
        features["operation_ids"],
        features["node_mask"],
        torch.cat(
            [
                features["global_features"],
                features["constraint_diagnostics"],
            ],
            dim=-1,
        ),
        features["action_features"],
        features["hard_mask"],
    )


def _forward(model: ConstraintStateEncoder, features: dict) -> dict:
    return model(*_model_input(features))


def _positive_mask(size: int, batch: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(size, size, dtype=torch.bool, device=device)
    base = torch.arange(batch, device=device)
    members = torch.stack([base, base + batch, base + 2 * batch])
    mask[
        members[:, None, :],
        members[None, :, :],
    ] = True
    return mask


def _train_epoch(
    model: ConstraintStateEncoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    features: dict,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    generator = torch.Generator().manual_seed(args.seed * 1000 + epoch)
    order = torch.randperm(
        features["legal"].shape[0],
        generator=generator,
    )
    metrics = defaultdict(float)
    batches = 0
    use_amp = features["legal"].device.type == "cuda"
    for start in range(0, len(order), args.batch_size):
        indices = order[start : start + args.batch_size].to(features["legal"].device)
        batch = _move(features, indices, features["legal"].device)
        if len(indices) < 2:
            continue
        with torch.autocast(
            device_type=features["legal"].device.type,
            enabled=use_amp,
        ):
            augmented = _augment(batch, generator)
            embeddings = [
                _forward(model, view)["embedding"]
                for view in (
                    batch,
                    augmented["rotated"],
                    augmented["permuted"],
                )
            ]
            stacked = torch.cat(embeddings, dim=0)
            positive = _positive_mask(
                len(stacked),
                len(indices),
                stacked.device,
            )
            output = _forward(model, batch)
            contrastive = multi_positive_infonce(
                stacked,
                positive,
                temperature=args.temperature,
            )
            violation = multi_label_violation_loss(
                output["violation_logits"],
                batch["violation_labels"],
            )
            energy = energy_ranking_loss(output["energy"], batch["legal"])
            legal = torch.nn.functional.binary_cross_entropy_with_logits(
                output["legal_logits"],
                batch["legal"].float(),
            )
            loss = (
                args.contrastive_weight * contrastive
                + violation
                + legal
                + 0.1 * energy
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        metrics["loss"] += float(loss.detach())
        metrics["contrastive"] += float(contrastive.detach())
        metrics["violation"] += float(violation.detach())
        metrics["energy"] += float(energy.detach())
        metrics["legal"] += float(legal.detach())
        batches += 1
    return {
        key: value / max(batches, 1)
        for key, value in metrics.items()
    }


@torch.no_grad()
def _predict(
    model: ConstraintStateEncoder,
    features: dict,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    scores = {"energy": [], "violation": [], "legal": []}
    for start in range(0, features["legal"].shape[0], batch_size):
        batch = _move(
            features,
            torch.arange(
                start,
                min(start + batch_size, features["legal"].shape[0]),
            ),
            features["legal"].device,
        )
        output = _forward(model, batch)
        scores["energy"].append(
            (-output["energy"]).sigmoid().cpu().numpy()
        )
        scores["violation"].append(
            (
                1.0
                - output["violation_logits"].sigmoid().amax(dim=-1)
            ).cpu().numpy()
        )
        scores["legal"].append(
            output["legal_logits"].sigmoid().cpu().numpy()
        )
    return {key: np.concatenate(value) for key, value in scores.items()}


def _score_report(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    predicted = scores >= threshold
    return {
        "auroc": binary_auroc(scores, labels),
        "brier": brier_score(scores, labels),
        "ece": expected_calibration_error(scores, labels),
        "false_feasible_rate": float(predicted[~labels].mean()),
        "true_feasible_rate": float(predicted[labels].mean()),
        "threshold": threshold,
        "rows": int(labels.size),
        "legal_rows": int(labels.sum()),
    }


def _stratified_reports(
    scores: np.ndarray,
    labels: np.ndarray,
    operations: np.ndarray,
    room_count: np.ndarray,
    threshold: float,
) -> dict:
    operation_report = {}
    for operation in sorted(set(operations.tolist())):
        mask = operations == operation
        if labels[mask].any() and (~labels[mask]).any():
            operation_report[str(operation)] = _score_report(
                scores[mask],
                labels[mask],
                threshold,
            )
    room_report = {}
    bins = (("1-4", 1, 4), ("5-7", 5, 7), ("8+", 8, 1000))
    for name, lower, upper in bins:
        mask = (room_count >= lower) & (room_count <= upper)
        if labels[mask].any() and (~labels[mask]).any():
            room_report[name] = _score_report(
                scores[mask],
                labels[mask],
                threshold,
            )
    return {"operation": operation_report, "room_count": room_report}


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    train = _load_features(args.feature_root / "train/features.pt")
    validation = _load_features(args.feature_root / "val/features.pt")
    device = torch.device(args.device)
    for payload in (train, validation):
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                payload[key] = value.to(device)
    calibration_mask = validation["group_index"] % 2 == 0
    evaluation_mask = ~calibration_mask
    calibration = _subset(validation, calibration_mask)
    evaluation = _subset(validation, evaluation_mask)
    model = ConstraintStateEncoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    best = {"auroc": -1.0, "epoch": 0, "state": None, "mode": "energy"}
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_epoch(
            model,
            optimizer,
            scaler,
            train,
            args,
            epoch,
        )
        scheduler.step()
        calibration_scores = _predict(model, calibration, args.batch_size)
        labels = calibration["legal"].cpu().numpy()
        aurocs = {
            mode: binary_auroc(score, labels)
            for mode, score in calibration_scores.items()
        }
        mode = "legal"
        history.append(
            {
                "epoch": epoch,
                **train_metrics,
                "calibration_auroc": aurocs,
            }
        )
        print(
            f"epoch={epoch} loss={train_metrics['loss']:.4f} "
            f"cal_auroc={aurocs[mode]:.4f}",
            flush=True,
        )
        if aurocs[mode] > best["auroc"]:
            best = {
                "auroc": aurocs[mode],
                "epoch": epoch,
                "state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "mode": mode,
            }
    model.load_state_dict(best["state"])
    calibration_scores = _predict(model, calibration, args.batch_size)[best["mode"]]
    calibration_labels = calibration["legal"].cpu().numpy()
    temperature = fit_temperature(calibration_scores, calibration_labels)
    calibrated_calibration = apply_temperature(
        calibration_scores,
        temperature,
    )
    threshold = select_threshold(calibrated_calibration, calibration_labels)
    evaluation_scores = _predict(model, evaluation, args.batch_size)[best["mode"]]
    evaluation_labels = evaluation["legal"].cpu().numpy()
    calibrated_evaluation = apply_temperature(
        evaluation_scores,
        temperature,
    )
    overall = _score_report(
        calibrated_evaluation,
        evaluation_labels,
        threshold["threshold"],
    )
    stratified = _stratified_reports(
        calibrated_evaluation,
        evaluation_labels,
        evaluation["operation_ids"].cpu().numpy(),
        evaluation["room_count"].cpu().numpy(),
        threshold["threshold"],
    )
    checks = {
        "validation_auroc_at_least_0_90": overall["auroc"] >= 0.90,
        "false_feasible_rate_at_most_0_05": (
            overall["false_feasible_rate"] <= 0.05
        ),
        "calibration_ece_at_most_0_05": overall["ece"] <= 0.05,
    }
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_root / f"seed{args.seed}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "node_features": 8,
                "hidden_dim": 384,
                "embedding_dim": 192,
                "layers": 6,
                "operations": 5,
                "families": 5,
                "violation_families": 7,
            },
            "score_mode": best["mode"],
            "threshold": threshold["threshold"],
            "seed": args.seed,
            "epoch": best["epoch"],
        },
        checkpoint_path,
    )
    report = {
        "schema_version": "residential-constraint-embedding-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only",
        "selection_policy": "train-only fit; odd/even validation group split",
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "score_mode": best["mode"],
        "temperature": temperature,
        "calibration": {
            **threshold,
            "rows": int(calibration_labels.size),
        },
        "evaluation": overall,
        "stratified": stratified,
        "checks": checks,
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(ROOT))
        if checkpoint_path.is_relative_to(ROOT)
        else str(checkpoint_path),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(args.report), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
